#!/usr/bin/env python3
# =============================================================================
# Estensione 6.1 VPR — Recall@1 adattiva con soglia fissa dal val (SF-XS)
#
# A differenza della curva sweep, qui il gate usa una soglia FISSA P*
# ottimizzata su F1 sul validation set (SF-XS val).
# Per ogni query:
#   P(top1_correct) >= P*  →  accetta retrieval top-1 (no re-ranking)
#   P(top1_correct) <  P*  →  usa reranked_correct
#
# Output:
#   adaptive_recall_val_threshold.csv   ← tabella riepilogativa
#   adaptive_recall_val_threshold.png   ← confronto bar chart
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA     = "/content/drive/MyDrive/Progetto_Machine_Learning"
FEAT_DIR = os.path.join(DATA, "feature_csv_6_1")
TRAIN_D  = os.path.join(FEAT_DIR, "features_train")
VAL_D    = os.path.join(FEAT_DIR, "features_val")
TEST_D   = os.path.join(FEAT_DIR, "features_test")
RERK_D   = os.path.join(FEAT_DIR, "features_test")  # reranked_debug_*.csv
OUT_DIR  = os.path.join(DATA, "results_6_1_alltests")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS  = ["megaloc", "netvlad"]
MATCHERS = ["superpoint-lg", "superglue"]
DATASETS = {"svox_sun": "sun", "svox_night": "night",
            "sf_xs": "sf_xs", "tokyo_xs": "tokyo_xs"}

FEAT_COLS = ["clustered_score", "relative_gap", "d1", "num_inliers"]
ALL_FEATS = ["d1", "d2", "gap", "relative_gap",
             "num_inliers", "inlier_ratio", "clustered_score"]
TARGET    = "top1_correct"
C_GRID    = [0.01, 0.1, 1.0, 10.0, 100.0]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def best_threshold(y, score):
    """Soglia che massimizza F1 sul val set."""
    y = np.asarray(y); score = np.asarray(score)
    cand = np.unique(np.quantile(score, np.linspace(0, 1, 301)))
    P = y.sum(); bt, bf = 0.5, -1.0
    for t in cand:
        pred = score >= t
        tp = np.logical_and(pred, y == 1).sum(); pp = pred.sum()
        prec = tp / pp if pp > 0 else 0.0
        rec  = tp / P  if P  > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f > bf:
            bf, bt = f, t
    return float(bt), float(bf)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

rows = []

for method in METHODS:
    for matcher in MATCHERS:
        key = f"{method}_{matcher}"

        # Percorsi train e val
        ftr = os.path.join(TRAIN_D, f"features_{method}_{matcher}_sun_night.csv")
        fva = os.path.join(VAL_D,   f"features_{method}_sf_xs_val_{matcher}.csv")
        if not (os.path.exists(ftr) and os.path.exists(fva)):
            print(f"[SKIP] manca train/val per {key}"); continue

        dtr = pd.read_csv(ftr); dva = pd.read_csv(fva)
        idx = {f: i for i, f in enumerate(ALL_FEATS)}
        cols = [idx[f] for f in FEAT_COLS]

        Xtr = dtr[ALL_FEATS].values; ytr = dtr[TARGET].astype(int).values
        Xva = dva[ALL_FEATS].values; yva = dva[TARGET].astype(int).values

        scaler = StandardScaler().fit(Xtr)
        Str = scaler.transform(Xtr)
        Sva = scaler.transform(Xva)

        # Seleziona C su val AUC
        bestC, bestAuc = 1.0, -1.0
        for C in C_GRID:
            clf_c = LogisticRegression(C=C, max_iter=1000).fit(Str[:, cols], ytr)
            va = roc_auc_score(yva, clf_c.predict_proba(Sva[:, cols])[:, 1]) \
                 if len(set(yva)) > 1 else np.nan
            if not np.isnan(va) and va > bestAuc:
                bestAuc, bestC = va, C

        # Addestra modello finale, calcola soglia F1 su val
        clf = LogisticRegression(C=bestC, max_iter=1000).fit(Str[:, cols], ytr)
        p_va = clf.predict_proba(Sva[:, cols])[:, 1]
        val_thr, val_f1 = best_threshold(yva, p_va)

        print(f"\n--- {key} ---")
        print(f"    C={bestC}  val_AUC={bestAuc:.4f}  val_thr={val_thr:.4f}  val_F1={val_f1:.4f}")

        # Applica al test
        for dlabel, tok in DATASETS.items():
            fte = os.path.join(TEST_D, f"features_{method}_{tok}_test_{matcher}.csv")
            # Fallback: alcuni file stanno in RERK_D
            if not os.path.exists(fte):
                fte = os.path.join(RERK_D, f"features_{method}_{tok}_test_{matcher}.csv")
            if not os.path.exists(fte):
                print(f"    [SKIP] feature test mancante: {dlabel}"); continue

            # Reranked correct — prova più path possibili
            rk_candidates = [
                os.path.join(RERK_D, f"reranked_debug_{method}_{tok}_{matcher}.csv"),
                os.path.join(RERK_D, f"reranked_correct_{method}_{tok}_{matcher}.csv"),
                os.path.join(RERK_D, f"reranked_debug_{method}_{dlabel}_{matcher}.csv"),
                os.path.join(RERK_D, f"reranked_correct_{method}_{dlabel}_{matcher}.csv"),
            ]
            rk_path = next((p for p in rk_candidates if os.path.exists(p)), None)

            # Fallback: cerca nel consolidato
            if rk_path is None:
                cons = os.path.join(RERK_D, "rerank_eval_consolidato.csv")
                if os.path.exists(cons):
                    df_cons = pd.read_csv(cons)
                    sub_cons = df_cons[
                        (df_cons["method"] == method) &
                        (df_cons["matcher"] == matcher) &
                        (df_cons["dataset"].isin([tok, dlabel]))
                    ]
                    if not sub_cons.empty:
                        rk_path = "__consolidato__"
                        drk_raw = sub_cons[["query_id", "reranked_correct"]].dropna()
                    else:
                        print(f"    [SKIP] reranked_correct mancante: {dlabel}"); continue
                else:
                    print(f"    [SKIP] reranked_correct mancante: {dlabel}"); continue

            dte = pd.read_csv(fte)
            if rk_path == "__consolidato__":
                drk = drk_raw.copy()
            else:
                drk = pd.read_csv(rk_path)[["query_id", "reranked_correct"]].dropna()
            drk["reranked_correct"] = drk["reranked_correct"].astype(int)
            drk["query_id"] = drk["query_id"].astype(int)
            dte["query_id_int"] = dte["query_id"].astype(int)

            merged = dte[["query_id_int", TARGET] + ALL_FEATS].merge(
                drk.rename(columns={"query_id": "query_id_int"}),
                on="query_id_int", how="inner"
            )
            if len(merged) < len(dte):
                print(f"    WARNING {dlabel}: {len(dte)-len(merged)} query non allineate")

            Ste = scaler.transform(merged[ALL_FEATS].values)
            p   = clf.predict_proba(Ste[:, cols])[:, 1]

            ret_ok  = merged[TARGET].values.astype(float)
            rerk_ok = merged["reranked_correct"].values.astype(float)

            # Gate: P >= val_thr → accetta retrieval; P < val_thr → usa reranked
            gate_mask = p < val_thr          # True = re-rank questa query
            adaptive  = np.where(gate_mask, rerk_ok, ret_ok)

            r1_ret      = float(ret_ok.mean())
            r1_full     = float(rerk_ok.mean())
            r1_adaptive = float(adaptive.mean())
            pct_reranked = float(gate_mask.mean()) * 100
            pct_skipped  = 100.0 - pct_reranked

            print(f"    {dlabel:9s}: ret={r1_ret*100:.1f}  "
                  f"full={r1_full*100:.1f}  "
                  f"adaptive={r1_adaptive*100:.1f}  "
                  f"re-ranked={pct_reranked:.1f}%  "
                  f"skipped={pct_skipped:.1f}%")

            rows.append(dict(
                case=key, method=method, matcher=matcher, dataset=dlabel,
                C=bestC, val_AUC=round(bestAuc, 4),
                val_threshold=round(val_thr, 4), val_F1=round(val_f1, 4),
                n_queries=len(merged),
                R1_retrieval=round(r1_ret * 100, 2),
                R1_full_rerank=round(r1_full * 100, 2),
                R1_adaptive=round(r1_adaptive * 100, 2),
                delta_adaptive_vs_retrieval=round((r1_adaptive - r1_ret) * 100, 2),
                delta_adaptive_vs_full=round((r1_adaptive - r1_full) * 100, 2),
                pct_reranked=round(pct_reranked, 1),
                pct_skipped=round(pct_skipped, 1),
            ))

# ── OUTPUT ────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
csv_out = os.path.join(OUT_DIR, "adaptive_recall_val_threshold.csv")

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== RIEPILOGO ===")
if df.empty:
    print("Nessun risultato: controlla che i file reranked_correct esistano in:")
    print(f"  {RERK_D}")
    print("File attesi: reranked_debug_<method>_<tok>_<matcher>.csv")
    print("             reranked_correct_<method>_<tok>_<matcher>.csv")
    print("             rerank_eval_consolidato.csv")
    import sys; sys.exit(0)

df.to_csv(csv_out, index=False)
print(df[["case", "dataset", "val_threshold",
          "R1_retrieval", "R1_full_rerank", "R1_adaptive",
          "pct_skipped"]].to_string(index=False))
print(f"\nSalvato: {csv_out}")

# ── PLOT ──────────────────────────────────────────────────────────────────────

cases   = df["case"].unique()
dslabels = list(DATASETS.keys())
n_rows  = len(cases)
n_cols  = len(dslabels)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                         squeeze=False)

for r, case in enumerate(cases):
    for c, ds in enumerate(dslabels):
        ax  = axes[r][c]
        sub = df[(df["case"] == case) & (df["dataset"] == ds)]
        if sub.empty:
            ax.set_visible(False); continue
        row = sub.iloc[0]

        vals  = [row["R1_retrieval"], row["R1_full_rerank"], row["R1_adaptive"]]
        clrs  = ["#4C72B0", "#C44E52", "#55A868"]
        bars  = ax.bar(["Retrieval", "Full re-rank", "Adaptive\n(val thr)"],
                       vals, color=clrs, width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{case}\n{ds}  (thr={row['val_threshold']:.3f}, "
                     f"skip={row['pct_skipped']:.0f}%)", fontsize=8)
        ax.set_ylabel("Recall@1 (%)")
        ax.set_ylim(0, min(105, max(vals) + 8))
        ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Adaptive Re-ranking — soglia fissa dal val SF-XS (F1-ottimizzata)\n"
    "Grigio=retrieval  Rosso=full re-ranking  Verde=adaptive",
    fontsize=10, y=1.01
)
plt.tight_layout()

png_out = os.path.join(OUT_DIR, "adaptive_recall_val_threshold.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight")
print(f"Plot salvato: {png_out}")