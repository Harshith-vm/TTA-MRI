# analysis/hypothesis_test.py
"""
Confirmatory statistics for the central hypothesis:

  H1: Explanation Stability (ESI) predicts post-TTA calibration degradation
      (ΔECE) OVER AND ABOVE any change in discriminative performance (ΔAUC).

Upgrades over the naive version (all reviewer-requested):
  - Linear Mixed-Effects Model with random intercepts for model & dataset
    (the same backbone appears many times -> rows are NOT independent).
  - PAIRED Wilcoxon of each TTA method vs. its matched no-TTA baseline.
  - Benjamini-Hochberg FDR *and* Bonferroni (report both).
  - Cliff's delta non-parametric effect size for every comparison.
  - Bootstrap 95% CIs on per-(model,method) mean ESI.
  - Friedman omnibus + Nemenyi post-hoc -> critical-difference ranking.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_all_results(results_jsonl: Path) -> pd.DataFrame:
    rec = []
    with open(results_jsonl) as f:
        for line in f:
            try:
                r = json.loads(line)
                rec.append({
                    "dataset": r["dataset"], "model": r["model"],
                    "tta_method": r["tta_method"], "domain": str(r["domain"]),
                    "seed": r["seed"],
                    "delta_auc": r["deltas"]["delta_auc"],
                    "delta_ece": r["deltas"]["delta_ece"],
                    "delta_brier": r["deltas"]["delta_brier"],
                    "esi_score": r["explanation_stability"]["mean_esi"],
                    "silent_failure": r["silent_failure"],
                    "pre_auc": r["pre_tta"]["auc"], "post_auc": r["post_tta"]["auc"],
                    "pre_ece": r["pre_tta"]["ece"], "post_ece": r["post_tta"]["ece"],
                })
            except Exception:
                continue
    return pd.DataFrame(rec)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (n * m)


def bootstrap_ci(x, n_boot=config.N_BOOTSTRAP, alpha=0.05):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)])
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return (float(np.mean(x)),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def run_full_analysis(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=['esi_score', 'delta_ece', 'delta_auc']).copy()
    print("=" * 64)
    print("H1: ESI predicts ΔECE independent of ΔAUC  (mixed-effects model)")
    print("=" * 64)

    # --- 1. Linear Mixed-Effects Model ---------------------------------------
    import statsmodels.formula.api as smf
    try:
        lmm = smf.mixedlm("delta_ece ~ esi_score + delta_auc", df,
                          groups=df["model"],
                          re_formula="~1",
                          vc_formula={"dataset": "0 + C(dataset)"}).fit(reml=True)
        with open(output_dir / "mixedlm_summary.txt", "w") as f:
            f.write(str(lmm.summary()))
        print(lmm.summary())
    except Exception as e:
        print(f"  [LMM] fell back to OLS: {e}")
        lmm = smf.ols("delta_ece ~ esi_score + delta_auc", df).fit()
        with open(output_dir / "ols_summary.txt", "w") as f:
            f.write(str(lmm.summary()))

    # --- 2. Partial correlation (Spearman, controlling ΔAUC) -----------------
    try:
        import pingouin as pg
        partial = pg.partial_corr(df, x='esi_score', y='delta_ece',
                                  covar='delta_auc', method='spearman')
        partial.to_csv(output_dir / "partial_correlation.csv")
        print("\nPartial corr (ESI~ΔECE | ΔAUC):\n", partial)
    except Exception as e:
        print(f"  [partial_corr] skipped: {e}")

    # --- 3. Paired Wilcoxon vs matched no-TTA baseline + FDR + Cliff's delta --
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    keys = ["dataset", "model", "domain", "seed"]
    base = df[df.tta_method == "no_tta"].set_index(keys)
    rows = []
    for method in [m for m in df.tta_method.unique() if m != "no_tta"]:
        sub = df[df.tta_method == method].set_index(keys)
        common = sub.index.intersection(base.index)
        for metric in ["delta_ece", "delta_auc", "esi_score"]:
            a = sub.loc[common, metric].values
            b = base.loc[common, metric].values
            if len(a) < 5:
                continue
            try:
                stat, p = wilcoxon(a, b)  # PAIRED
            except Exception:
                stat, p = np.nan, 1.0
            rows.append({"method": method, "metric": metric, "n_pairs": len(a),
                         "statistic": stat, "p_value": p,
                         "cliffs_delta": cliffs_delta(a, b),
                         "median_diff": float(np.median(a - b))})
    wr = pd.DataFrame(rows)
    if not wr.empty:
        wr["p_bonferroni"] = multipletests(wr.p_value, method="bonferroni")[1]
        wr["p_fdr_bh"] = multipletests(wr.p_value, method="fdr_bh")[1]
        wr["sig_fdr"] = wr.p_fdr_bh < 0.05
        wr.to_csv(output_dir / "paired_wilcoxon.csv", index=False)
        print("\nPaired Wilcoxon (vs no_tta) w/ FDR saved.")

    # --- 4. Bootstrap CIs on mean ESI per (model, method) --------------------
    ci_rows = []
    for (mdl, mth), g in df.groupby(["model", "tta_method"]):
        mean, lo, hi = bootstrap_ci(g.esi_score.values)
        ci_rows.append({"model": mdl, "tta_method": mth,
                        "mean_esi": mean, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ci_rows).to_csv(output_dir / "esi_bootstrap_ci.csv", index=False)

    # --- 5. Friedman + Nemenyi (critical-difference ranking) -----------------
    friedman_nemenyi(df, "esi_score", output_dir, "esi")
    friedman_nemenyi(df, "delta_ece", output_dir, "delta_ece")

    # --- 6. Silent-failure prevalence ----------------------------------------
    sf = df.groupby("tta_method")["silent_failure"].mean().sort_values(ascending=False)
    sf.to_csv(output_dir / "silent_failure_rates.csv")
    print("\nSilent-failure rate by method:\n", sf)
    print(f"\nAll analysis written to {output_dir}")


def friedman_nemenyi(df, metric, output_dir, tag):
    """Friedman omnibus across TTA methods (blocks = model×dataset×domain×seed)."""
    from scipy.stats import friedmanchisquare
    keys = ["dataset", "model", "domain", "seed"]
    piv = df.pivot_table(index=keys, columns="tta_method", values=metric)
    piv = piv.dropna(axis=0, how="any")
    if piv.shape[0] < 5 or piv.shape[1] < 3:
        print(f"  [Friedman:{tag}] insufficient complete blocks.")
        return
    stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
    out = {"metric": metric, "friedman_chi2": float(stat), "p_value": float(p),
           "n_blocks": int(piv.shape[0]), "mean_ranks": {}}
    ranks = piv.rank(axis=1, ascending=(metric != "esi_score"))  # ESI higher=better
    out["mean_ranks"] = ranks.mean().sort_values().to_dict()
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = nem.columns = piv.columns
        nem.to_csv(output_dir / f"nemenyi_{tag}.csv")
    except Exception as e:
        print(f"  [Nemenyi:{tag}] skipped: {e}")
    with open(output_dir / f"friedman_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [Friedman:{tag}] chi2={stat:.2f} p={p:.2e}  ranks saved.")


if __name__ == "__main__":
    res = config.RESULTS_JSONL
    if res.exists():
        run_full_analysis(load_all_results(res), config.AGG_ROOT)
    else:
        print(f"Results file {res} not found.")
