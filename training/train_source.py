# training/train_source.py
"""
Trains each model on a source domain. Two source datasets supported:
  --dataset camelyon17  -> PCam (binary)        [default]
  --dataset nct_crc     -> NCT-CRC-HE-100K (9-class)

Checkpoints saved every epoch; training resumes from last checkpoint if
interrupted. BF16 mixed precision (Ampere/A6000), modern torch.amp API.

Usage: python training/train_source.py --model resnet50 --seed 17 --dataset camelyon17
"""
import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.amp import GradScaler, autocast          # modern API (was torch.cuda.amp)
import config
from models.model_registry import get_model
from models.transforms import get_transforms
from utils.io_utils import atomic_torch_save, atomic_json_save, load_json_safe
from utils.registry import init_registry, mark_running, mark_done, mark_failed, is_done

CKPT_ROOT = config.CKPT_ROOT / "source_trained"


def _source_datasets(dataset, model_name):
    if dataset == "nct_crc":
        from data.nct_crc_dataset import NCTCRCDataset
        full = NCTCRCDataset(split="train", transform=get_transforms(model_name, 'train'))
        n_val = len(full) // 10
        g = torch.Generator().manual_seed(0)
        val_idx = set(torch.randperm(len(full), generator=g)[:n_val].tolist())
        tr = torch.utils.data.Subset(full, [i for i in range(len(full)) if i not in val_idx])
        va_full = NCTCRCDataset(split="train", transform=get_transforms(model_name, 'val'))
        va = torch.utils.data.Subset(va_full, sorted(val_idx))
        return tr, va, 9
    else:
        from data.pcam_dataset import PCamDataset
        tr = PCamDataset(split='train', transform=get_transforms(model_name, 'train'))
        va = PCamDataset(split='valid', transform=get_transforms(model_name, 'val'))
        return tr, va, 2


def train_epoch(model, loader, optimizer, scaler, device, n_classes):
    model.train()
    total_loss, correct, total = 0, 0, 0
    crit = nn.CrossEntropyLoss()
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=config.AMP_DTYPE):
            logits = model(imgs)
            loss = crit(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}", flush=True)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device, n_classes):
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type='cuda', dtype=config.AMP_DTYPE):
            logits = model(imgs)
        probs = torch.softmax(logits.float(), dim=1)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_probs.append(probs.cpu()); all_labels.extend(labels.cpu().tolist())
    from sklearn.metrics import roc_auc_score
    import numpy as np
    P = torch.cat(all_probs).numpy(); y = np.array(all_labels)
    if n_classes == 2:
        auc = roc_auc_score(y, P[:, 1])
    else:
        auc = roc_auc_score(y, P, multi_class='ovr', average='macro')
    return correct / total, float(auc)


def train_model(model_name, seed, dataset="camelyon17", n_epochs=30,
                patience=5, min_epochs=5):
    init_registry()
    stage = f"source_training_{dataset}"
    ckpt_dir = CKPT_ROOT / dataset / model_name / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / "best_model.pt"
    if final_ckpt.exists():
        print(f"[SKIP] {model_name} {dataset} seed={seed} already trained.")
        return final_ckpt

    mark_running(model_name, "none", -1, seed, stage)
    torch.manual_seed(seed)
    device = config.DEVICE
    print(f"\n=== Training {model_name} | {dataset} | seed={seed} | {device} ===")

    train_ds, val_ds, n_classes = _source_datasets(dataset, model_name)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, sampler=get_balanced_sampler(train_ds, n_classes),
        persistent_workers=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=False, persistent_workers=False)

    model, feat_dim, norm_type = get_model(model_name, num_classes=n_classes)
    if hasattr(model, 'unfreeze_last_n_blocks'):
        model.unfreeze_last_n_blocks(n=2)
    model = model.to(device)
    if config.USE_COMPILE and model_name in ("resnet50", "efficientnet_b3"):
        model = torch.compile(model, mode="reduce-overhead")

    lr = 1e-3 if model_name in ('phikon_v2', 'h_optimus_0', 'prov_gigapath') else 1e-4
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    start_epoch, best_auc = 0, 0.0
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if epoch_ckpts:
        state = torch.load(epoch_ckpts[-1], map_location=device)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")

    history = load_json_safe(ckpt_dir / "history.json", default=[])
    no_improve = 0
    try:
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, device, n_classes)
            va_acc, va_auc = eval_epoch(model, val_loader, device, n_classes)
            scheduler.step()
            dt = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{n_epochs} | loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_acc={va_acc:.4f} val_auc={va_auc:.4f} | {dt:.0f}s")
            atomic_torch_save({'epoch': epoch, 'model': model.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict(),
                               'scaler': scaler.state_dict(), 'val_auc': va_auc},
                              ckpt_dir / f"epoch_{epoch:03d}.pt")
            # Early stopping: count epochs without a meaningful val_auc gain.
            if va_auc > best_auc + 1e-4:
                best_auc = va_auc
                no_improve = 0
                atomic_torch_save({'epoch': epoch, 'model': model.state_dict(),
                                   'model_name': model_name, 'seed': seed,
                                   'dataset': dataset, 'num_classes': n_classes,
                                   'val_auc': va_auc, 'val_acc': va_acc}, final_ckpt)
                print(f"  ** new best (AUC={va_auc:.4f})")
            else:
                no_improve += 1
            for old in sorted(ckpt_dir.glob("epoch_*.pt"))[:-3]:
                old.unlink()
            history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                            'val_acc': va_acc, 'val_auc': va_auc, 'elapsed_s': dt})
            atomic_json_save(history, ckpt_dir / "history.json")
            if epoch + 1 >= min_epochs and no_improve >= patience:
                print(f"  early stop: no val_auc gain for {patience} epochs "
                      f"(best={best_auc:.4f} @ epoch saved). Saturated dataset.")
                break
    except Exception as e:
        mark_failed(model_name, "none", -1, seed, stage, str(e)); raise

    mark_done(model_name, "none", -1, seed, stage, result_path=str(final_ckpt))
    print(f"\nDone. Best AUC={best_auc:.4f}. -> {final_ckpt}")
    return final_ckpt


def _fast_labels(dataset):
    """Read integer labels WITHOUT decoding images (fast, no open file handles)."""
    import numpy as np
    # PCam: read straight from the y HDF5
    if hasattr(dataset, "get_all_labels"):
        return np.asarray(dataset.get_all_labels())
    # NCT-CRC raw dataset: labels live in .samples
    if hasattr(dataset, "samples"):
        return np.asarray([lbl for _, lbl in dataset.samples])
    # torch Subset wrapping one of the above
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        if hasattr(base, "get_all_labels"):
            return np.asarray(base.get_all_labels())[np.asarray(dataset.indices)]
        if hasattr(base, "samples"):
            return np.asarray([base.samples[i][1] for i in dataset.indices])
    # Fallback (slow): iterate
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def get_balanced_sampler(dataset, n_classes):
    labels = torch.as_tensor(_fast_labels(dataset), dtype=torch.long)
    counts = torch.bincount(labels, minlength=n_classes).float()
    weights = 1.0 / counts.clamp(min=1)
    sw = weights[labels]
    return torch.utils.data.WeightedRandomSampler(sw, len(sw))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dataset", default="camelyon17", choices=["camelyon17", "nct_crc"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    train_model(args.model, args.seed, args.dataset, args.epochs)
