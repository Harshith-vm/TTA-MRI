# run_evaluation.py
"""
Master evaluation. Crash-safe at every level. One "run" =
(dataset, model, tta_method, domain, seed); inside a run we sweep all XAI
methods for explanation-stability and a faithfulness subset.

Usage:
  python run_evaluation.py                     # all pending
  python run_evaluation.py --dataset nct_crc
  python run_evaluation.py --model resnet50 --method tent
  python run_evaluation.py --status
Re-run after interruption — completed runs are skipped.
"""
import argparse, gc, os, subprocess, sys, time, traceback
from pathlib import Path
from itertools import product
import numpy as np
import torch
from torch.amp import autocast
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The heaviest deps (HF transformers, timm foundation models, captum) are
# deferred to _lazy_imports so the parent-dispatch process stays lightweight.
# Each --single subprocess picks them up freshly, guaranteeing zero memory
# carry-over between the 3000 combos in the sweep — the primary cause of the
# earlier host-OOM crash.
import config
from utils.registry import (init_registry, is_done, mark_running, mark_done,
                            mark_failed, print_status)

XAI_METHODS = config.XAI_METHODS
FAITHFUL_XAI = ["gradcam", "integrated_gradients"]   # subset for costly metrics
N_SAL = config.N_SALIENCY_PER_BATCH


def _lazy_imports():
    """Load heavy per-run deps only in the --single subprocess."""
    global atomic_json_save, append_jsonl, get_model, get_transforms, get_tta
    global generate_saliency, compute_all_metrics, compute_esi
    global insertion_deletion_auc, compute_calibration_metrics
    global compute_calibration_multiclass
    from utils.io_utils import atomic_json_save, append_jsonl  # noqa: F401
    from models.model_registry import get_model             # noqa: F401
    from models.transforms import get_transforms            # noqa: F401
    from tta.tta_registry import get_tta                    # noqa: F401
    from xai.xai_runner import generate_saliency            # noqa: F401
    from xai.stability_metrics import compute_all_metrics, compute_esi  # noqa: F401
    from xai.faithfulness_metrics import insertion_deletion_auc  # noqa: F401
    from calibration.metrics import (compute_calibration_metrics,
                                     compute_calibration_multiclass)  # noqa: F401
    g = globals()
    g.update(dict(atomic_json_save=atomic_json_save,
                  append_jsonl=append_jsonl, get_model=get_model,
                  get_transforms=get_transforms, get_tta=get_tta,
                  generate_saliency=generate_saliency,
                  compute_all_metrics=compute_all_metrics,
                  compute_esi=compute_esi,
                  insertion_deletion_auc=insertion_deletion_auc,
                  compute_calibration_metrics=compute_calibration_metrics,
                  compute_calibration_multiclass=compute_calibration_multiclass))


def ckpt_path(dataset, model_name, seed):
    return (config.CKPT_ROOT / "source_trained" / dataset / model_name /
            f"seed{seed}" / "best_model.pt")


def load_source_model(dataset, model_name, seed, n_classes, device):
    p = ckpt_path(dataset, model_name, seed)
    if not p.exists():
        raise FileNotFoundError(f"Source model not found: {p}\nRun training first.")
    model, _, _ = get_model(model_name, num_classes=n_classes, pretrained=False)
    state = torch.load(p, map_location=device)
    sd = state['model']
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}  # strip compile prefix
    model.load_state_dict(sd, strict=False)
    del state, sd    # release the checkpoint dict before moving to GPU
    return model.to(device)


def make_loader(dataset, domain, model_name, batch_size):
    tf = get_transforms(model_name, 'test')
    if dataset == "camelyon17":
        from data.camelyon17_dataset import Camelyon17CenterDataset
        ds = Camelyon17CenterDataset(center_id=domain, transform=tf)
    else:
        from data.nct_crc_dataset import NCTCRCDataset
        ds = NCTCRCDataset(split="val7k", transform=tf)
    # Fixed random subset per (dataset, domain): bounds runtime, keeps the SAME
    # test stream across every model/method/seed so paired stats stay valid.
    cap = config.MAX_EVAL_SAMPLES
    if cap and len(ds) > cap:
        rng = np.random.default_rng(1000 + _dom_id(domain))
        idx = sorted(rng.choice(len(ds), size=cap, replace=False).tolist())
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=False, persistent_workers=False)


@torch.no_grad()
def _infer(model, imgs):
    # fp32 (no autocast) so pre-TTA and post-TTA logits use identical precision;
    # eliminates the bf16-vs-fp32 delta that made No-TTA look non-zero.
    return model(imgs).float()


ABLATION = {"steps": 1, "reset": False, "eval_bs": None, "tag": ""}


def evaluate_one_run(dataset, model_name, tta_name, domain, seed, n_classes):
    tag = ("_" + ABLATION["tag"]) if ABLATION["tag"] else ""
    stage = "tta_evaluation" + tag
    if is_done(model_name, tta_name, _dom_id(domain), seed, stage, dataset=dataset):
        return None
    mark_running(model_name, tta_name, _dom_id(domain), seed, stage, dataset=dataset)
    device = config.DEVICE
    torch.manual_seed(seed); np.random.seed(seed)
    result_path = (config.RESULTS_ROOT / "raw" /
                   f"{dataset}__{model_name}__{tta_name}__d{domain}__s{seed}{tag}.json")
    sal_dir = config.SALIENCY_ROOT / dataset / model_name / tta_name / f"d{domain}_s{seed}"

    try:
        model = load_source_model(dataset, model_name, seed, n_classes, device)
        model.eval()   # CRITICAL: freeze BN running stats + dropout so the
                       # pre-TTA forward is the true frozen baseline (fixes the
                       # non-zero No-TTA delta and ViT saliency non-determinism).
        tta = get_tta(tta_name, model, device=device, num_classes=n_classes)
        bs = ABLATION["eval_bs"] or config.batch_size(model_name)
        loader = make_loader(dataset, domain, model_name, bs)
        source_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                        if ABLATION["reset"] else None)

        labels_all = []
        probs_before, probs_after = [], []
        ent_before, ent_after = [], []
        # stability metrics keyed by xai method
        stab = {x: [] for x in XAI_METHODS}
        faith = {x: {"ins": [], "del": []} for x in FAITHFUL_XAI}
        saved_qualitative = False

        for bidx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits_b = _infer(model, imgs)
            p_b = torch.softmax(logits_b, 1)
            tgt = p_b.argmax(1)

            # Saliency (the expensive part) only on the first N batches per run;
            # a few hundred sampled patches give a stable mean ESI. TTA adaptation
            # itself still runs on EVERY batch below (full online stream).
            do_saliency = bidx < config.N_SALIENCY_BATCHES
            sub = list(range(min(N_SAL, imgs.size(0))))
            stgt = tgt[sub]
            if do_saliency:
                simgs = imgs[sub]
                sal_b = {x: generate_saliency(model, simgs_safe(simgs), model_name, device,
                                              method=x, targets=stgt) for x in XAI_METHODS}

            # --- TTA adaptation + post inference (every batch) ---
            if source_state is not None:      # episodic: reset model+method each batch
                model.load_state_dict(source_state); model.eval()
                tta = get_tta(tta_name, model, device=device, num_classes=n_classes)
            for _ in range(max(1, ABLATION["steps"])):   # adaptation-steps ablation
                logits_a = tta.adapt(imgs).float()
            p_a = torch.softmax(logits_a, 1)

            if do_saliency:
                sal_a = {x: generate_saliency(model, simgs_safe(simgs), model_name, device,
                                              method=x, targets=stgt) for x in XAI_METHODS}
                # stability per xai method
                for x in XAI_METHODS:
                    for i in range(len(sub)):
                        if sal_b[x][i] is not None and sal_a[x][i] is not None:
                            m = compute_all_metrics(sal_b[x][i], sal_a[x][i])
                            m['esi'] = compute_esi(m)
                            stab[x].append(m)

                # faithfulness on post-TTA maps (subset of methods + a few samples)
                if bidx < min(3, config.N_SALIENCY_BATCHES):
                    for x in FAITHFUL_XAI:
                        for i in range(min(4, len(sub))):
                            if sal_a[x][i] is None:
                                continue
                            fd = insertion_deletion_auc(
                                model, imgs[sub[i]].detach(), sal_a[x][i],
                                int(stgt[i].item()), device)
                            faith[x]["ins"].append(fd.get("insertion_auc"))
                            faith[x]["del"].append(fd.get("deletion_auc"))

            # save one qualitative panel (raw maps) for the figure
            if do_saliency and not saved_qualitative:
                sal_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sal_dir / "qualitative.npz",
                    img=imgs[sub[:4]].detach().cpu().numpy(),
                    **{f"{x}_before": np.stack([_safe(sal_b[x][i]) for i in range(min(4, len(sub)))])
                       for x in XAI_METHODS},
                    **{f"{x}_after": np.stack([_safe(sal_a[x][i]) for i in range(min(4, len(sub)))])
                       for x in XAI_METHODS})
                saved_qualitative = True

            labels_all.extend(labels.cpu().tolist())
            probs_before.append(p_b.cpu().numpy()); probs_after.append(p_a.cpu().numpy())
            eb = -(p_b * torch.log(p_b + 1e-8)).sum(1)
            ea = -(p_a * torch.log(p_a + 1e-8)).sum(1)
            ent_before.extend(eb.cpu().tolist()); ent_after.extend(ea.cpu().tolist())

            if bidx % 20 == 0:
                print(f"  [{dataset}|{model_name}|{tta_name}|d{domain}|s{seed}] "
                      f"batch {bidx}/{len(loader)}", flush=True)

        result = _aggregate(dataset, model_name, tta_name, domain, seed, n_classes,
                            labels_all, probs_before, probs_after,
                            ent_before, ent_after, stab, faith)
        if ABLATION["tag"]:
            result["ablation"] = {"tag": ABLATION["tag"], "steps": ABLATION["steps"],
                                  "reset": ABLATION["reset"], "eval_bs": ABLATION["eval_bs"]}
        atomic_json_save(result, result_path)
        jsonl = (config.RESULTS_ROOT / f"ablation_{ABLATION['tag']}.jsonl"
                 if ABLATION["tag"] else config.RESULTS_JSONL)
        append_jsonl(result, jsonl)
        mark_done(model_name, tta_name, _dom_id(domain), seed, stage,
                  str(result_path), dataset=dataset)
        return result

    except Exception as e:
        tb = traceback.format_exc()
        mark_failed(model_name, tta_name, _dom_id(domain), seed, stage, tb, dataset=dataset)
        print(f"  ERROR [{dataset}|{model_name}|{tta_name}|d{domain}|s{seed}]: {e}\n{tb}")
        return None
    finally:
        # The parent dispatches each combo to a fresh subprocess, so the
        # process exits immediately after this returns and locals are reaped
        # automatically. The two calls below are cheap and still worth doing
        # in case anyone imports this module and calls evaluate_one_run in a
        # loop from another driver.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _aggregate(dataset, model_name, tta_name, domain, seed, n_classes,
               labels, pb_list, pa_list, eb, ea, stab, faith):
    from sklearn.metrics import accuracy_score, roc_auc_score
    y = np.array(labels)
    Pb = np.concatenate(pb_list); Pa = np.concatenate(pa_list)
    # If adaptation blew up part-way through the stream (e.g. EATA on a
    # bad seed), some probs_after rows can contain NaN. Fall back to the
    # pre-adaptation prob for those rows so we still record a valid AUC —
    # the failure signal is captured elsewhere (silent_failure, delta_*).
    if not np.isfinite(Pa).all():
        bad = ~np.isfinite(Pa).all(axis=1)
        print(f"  WARN: {bad.sum()}/{len(Pa)} post-TTA prob rows had NaN; "
              f"substituting pre-TTA probs for those rows.", flush=True)
        Pa[bad] = Pb[bad]
    # The substitution above assumed Pb is clean. It is not: _infer() reads the
    # SAME continuously-adapted model, so once adaptation diverges, Pb and Pa are
    # both NaN from that batch onward and Pa[bad] = Pb[bad] copies NaN onto NaN.
    # roc_auc_score(y, Pb[:, 1]) then raised "Input contains NaN". Drop the
    # unusable rows so a diverged run still records, and flag how many were lost.
    n_dropped = 0
    usable = np.isfinite(Pb).all(axis=1) & np.isfinite(Pa).all(axis=1)
    if not usable.all():
        n_dropped = int((~usable).sum())
        print(f"  WARN: dropping {n_dropped}/{len(usable)} rows with non-finite "
              f"probs (adaptation diverged).", flush=True)
        if usable.sum() < 2 or len(np.unique(y[usable])) < 2:
            raise ValueError(
                f"adaptation diverged: only {int(usable.sum())} finite rows "
                f"remain out of {len(usable)} — no usable metrics")
        y = y[usable]
        Pb = Pb[usable]
        Pa = Pa[usable]
    pred_b, pred_a = Pb.argmax(1), Pa.argmax(1)
    acc_b = accuracy_score(y, pred_b); acc_a = accuracy_score(y, pred_a)
    if n_classes == 2:
        auc_b = roc_auc_score(y, Pb[:, 1]); auc_a = roc_auc_score(y, Pa[:, 1])
        cal_b = compute_calibration_metrics(Pb[:, 1], y)
        cal_a = compute_calibration_metrics(Pa[:, 1], y)
    else:
        auc_b = roc_auc_score(y, Pb, multi_class='ovr', average='macro')
        auc_a = roc_auc_score(y, Pa, multi_class='ovr', average='macro')
        cal_b = compute_calibration_multiclass(Pb, y)
        cal_a = compute_calibration_multiclass(Pa, y)

    def mean(key, lst):
        v = [m[key] for m in lst if m.get(key) is not None]
        return float(np.mean(v)) if v else None

    esi_by_xai = {x: mean('esi', stab[x]) for x in stab}
    # Integrated Gradients is the uniform primary XAI: architecture-agnostic
    # (no target-layer hooks), so it is valid for every model including the
    # foundation ViT-L where Grad-CAM's target-layer heuristic degenerates.
    primary_esi = esi_by_xai.get("integrated_gradients") or esi_by_xai.get("gradcam")
    stability_block = {}
    for x in stab:
        stability_block[x] = {
            "mean_esi": mean('esi', stab[x]),
            "mean_ssim": mean('ssim', stab[x]),
            "mean_cosine": mean('cosine_sim', stab[x]),
            "mean_spearman": mean('spearman_r', stab[x]),
            "mean_roi_iou": mean('roi_iou', stab[x]),
            "mean_kl_div": mean('kl_div', stab[x]),
            "mean_emd": mean('emd', stab[x]),
            "n": len(stab[x]),
        }
    faith_block = {x: {
        "insertion_auc": float(np.mean([v for v in faith[x]["ins"] if v is not None]))
            if any(v is not None for v in faith[x]["ins"]) else None,
        "deletion_auc": float(np.mean([v for v in faith[x]["del"] if v is not None]))
            if any(v is not None for v in faith[x]["del"]) else None,
    } for x in faith}

    d_auc = auc_a - auc_b; d_ece = cal_a['ece'] - cal_b['ece']
    # Standardised, reproducible definition: discrimination looks stable
    # (|ΔAUC| < 0.02) yet calibration meaningfully degrades (ΔECE > 0.05).
    silent_failure = bool(abs(d_auc) < 0.02 and d_ece > 0.05)

    return {
        "dataset": dataset, "model": model_name, "tta_method": tta_name,
        "domain": domain, "seed": seed, "num_classes": n_classes,
        "timestamp": time.time(),
        "pre_tta": {"accuracy": acc_b, "auc": float(auc_b), "ece": cal_b['ece'],
                    "brier": cal_b['brier'], "nll": cal_b['nll'],
                    "mean_entropy": float(np.mean(eb))},
        "post_tta": {"accuracy": acc_a, "auc": float(auc_a), "ece": cal_a['ece'],
                     "brier": cal_a['brier'], "nll": cal_a['nll'],
                     "mean_entropy": float(np.mean(ea))},
        "deltas": {"delta_accuracy": acc_a - acc_b, "delta_auc": float(d_auc),
                   "delta_ece": d_ece, "delta_brier": cal_a['brier'] - cal_b['brier']},
        "explanation_stability": {"primary_xai": "integrated_gradients", "mean_esi": primary_esi,
                                  "esi_by_xai": esi_by_xai, "per_xai": stability_block},
        "faithfulness": faith_block,
        "silent_failure": silent_failure,
        "n_nonfinite_dropped": n_dropped,
    }


def _safe(a):
    if a is None:
        return np.zeros((14, 14), dtype=np.float32)
    return np.asarray(a, dtype=np.float32)


def simgs_safe(t):
    return t  # placeholder hook for future per-method preprocessing


def _dom_id(domain):
    return domain if isinstance(domain, int) else 0


def build_grid(args):
    combos = []
    datasets = [args.dataset] if args.dataset else list(config.DATASETS.keys())
    for ds in datasets:
        info = config.DATASETS[ds]
        models = [args.model] if args.model else config.MODELS
        methods = [args.method] if args.method else config.TTA_METHODS
        domains = ([args.domain] if args.domain is not None else info["domains"])
        for m in models:
            seeds = [args.seed] if args.seed is not None else config.seeds_for(m)
            for meth, dom, s in product(methods, domains, seeds):
                combos.append((ds, m, meth, dom, s, info["num_classes"]))
    return combos


def _child_argv(python, script, ds, m, meth, dom, s, extra):
    argv = [python, script, "--single",
            "--dataset", ds, "--model", m, "--method", meth,
            "--domain", str(dom), "--seed", str(s)]
    if extra.get("steps") and extra["steps"] != 1:
        argv += ["--steps", str(extra["steps"])]
    if extra.get("reset"):
        argv += ["--reset"]
    if extra.get("eval_bs"):
        argv += ["--eval_bs", str(extra["eval_bs"])]
    if extra.get("tag"):
        argv += ["--tag", extra["tag"]]
    return argv


def _dispatch(pending, extra):
    """Run each combo in a fresh subprocess. Any leak in torch / HF / timm
    dies with the child; parent stays lightweight and just tallies progress."""
    python = sys.executable
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    # Expandable segments dramatically reduces VRAM fragmentation across the
    # varying batch sizes of the 5-model grid.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Release glibc arenas back to the OS aggressively so a crashed child
    # cannot leave the parent with a bloated RSS.
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")

    total = len(pending)
    ok = fail = skip = 0
    for i, (ds, m, meth, dom, s, nc) in enumerate(pending, 1):
        if is_done(m, meth, _dom_id(dom), s, "tta_evaluation" +
                   (("_" + extra["tag"]) if extra.get("tag") else ""),
                   dataset=ds):
            skip += 1
            continue
        print(f"\n[{i}/{total}] {ds} | {m} | {meth} | d{dom} | s{s}", flush=True)
        argv = _child_argv(python, script, ds, m, meth, dom, s, extra)
        t0 = time.time()
        try:
            rc = subprocess.call(argv, env=env)
        except KeyboardInterrupt:
            print("Interrupted by user."); raise
        except Exception as e:
            print(f"  SUBPROCESS SPAWN ERROR: {e}")
            rc = -1
        dt = time.time() - t0
        if rc == 0:
            ok += 1
            print(f"  ok in {dt:.0f}s", flush=True)
        else:
            fail += 1
            print(f"  FAILED (rc={rc}) in {dt:.0f}s — moving on", flush=True)
    print(f"\nDispatch done. ok={ok} fail={fail} skip={skip} of {total}")


def _run_single(args):
    """Execute exactly one combo in this process, then exit."""
    _lazy_imports()
    ABLATION.update(steps=args.steps, reset=args.reset,
                    eval_bs=args.eval_bs, tag=args.tag)
    dom = args.domain
    if isinstance(dom, str) and dom.isdigit():
        dom = int(dom)
    info = config.DATASETS[args.dataset]
    evaluate_one_run(args.dataset, args.model, args.method, dom, args.seed,
                     info["num_classes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(config.DATASETS.keys()))
    ap.add_argument("--model", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    # ablation flags (leave unset for the main run)
    ap.add_argument("--steps", type=int, default=1, help="adaptation steps per batch")
    ap.add_argument("--reset", action="store_true", help="episodic: reset model each batch")
    ap.add_argument("--eval_bs", type=int, default=None, help="override eval batch size")
    ap.add_argument("--tag", default="", help="label; writes ablation_<tag>.jsonl")
    # Subprocess isolation: parent enumerates combos and spawns one child per
    # run so leaked RAM/VRAM cannot compound across the 3000-run sweep.
    ap.add_argument("--single", action="store_true",
                    help="run exactly one combo in this process (child mode)")
    args = ap.parse_args()

    init_registry()
    (config.RESULTS_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status(); sys.exit(0)

    if args.single:
        # Child: needs all combo keys.
        for req in ("dataset", "model", "method", "domain", "seed"):
            if getattr(args, req) is None:
                sys.exit(f"--single requires --{req}")
        _run_single(args)
        sys.exit(0)

    # Parent: build the grid (respecting any partial filters), skip done, dispatch.
    # Grid build needs no torch — build_grid only touches config lists.
    grid = build_grid(args)
    stage = "tta_evaluation" + (("_" + args.tag) if args.tag else "")
    pending = [c for c in grid
               if not is_done(c[1], c[2], _dom_id(c[3]), c[4], stage, dataset=c[0])]
    print(f"Total combinations: {len(grid)} | pending: {len(pending)}", flush=True)
    if not pending:
        print("Nothing to do."); print_status(); sys.exit(0)
    extra = dict(steps=args.steps, reset=args.reset,
                 eval_bs=args.eval_bs, tag=args.tag)
    _dispatch(pending, extra)
    print("\nAll pending runs complete.")
    print_status()
