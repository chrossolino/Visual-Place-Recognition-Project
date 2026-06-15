#!/usr/bin/env python3
# =============================================================================
# Estensione 6.1 VPR — Curva Recall@1 vs % query attivate (adaptive re-ranking)
#
# Usa i coefficienti già calcolati (coefficients.csv) — nessun re-training.
# Ricalcola solo lo StandardScaler (media/std) sui dati di train, che è
# puramente descrittivo e deterministico.
#
# Input richiesti su Drive:
#   feature_csv_6_1/
#     features_<method>_<matcher>_sun_night.csv      ← train SVOX Sun+Night uniti
#     test/features_<method>_<ds>_test_<matcher>.csv ← test
#     test/reranked_debug_<method>_<ds>_<matcher>.csv ← R@1 post-reranking
#   coefficients.csv   ← caricato da Colab o da Drive (vedi COEFF_CSV)
# =============================================================================

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── CONFIG ────────────────────────────────────────────────────────────────────
DRIVE    = "/content/drive/MyDrive/Progetto_Machine_Learning"
FEAT_DIR = f"{DRIVE}/feature_csv_6_1"
TEST_DIR = f"{FEAT_DIR}/test"
OUT_DIR  = TEST_DIR

CONSOLIDATO = f"{TEST_DIR}/rerank_eval_consolidato.csv"

# Percorso del CSV dei coefficienti (carica da Drive o dalla sessione Colab)
COEFF_CSV = f"{DRIVE}/coefficients.csv"         # cambia se è altrove

METHODS  = ["megaloc", "netvlad"]
MATCHERS = ["superpoint-lg", "superglue"]

TEST_DSS  = ["sun", "night", "sf_xs", "tokyo_xs"]

FEAT_COLS = ["clustered_score", "relative_gap", "d1", "num_inliers"]
LABEL_COL = "top1_correct"

# ── STEP 1: carica coefficienti e ricostruisce modelli ────────────────────────

print("\n=== Carico coefficienti da", COEFF_CSV, "===")
df_coef = pd.read_csv(COEFF_CSV)

def load_logreg_from_coef(method, matcher):
    """
    Ricostruisce il modello 'completo' per (method, matcher) dai coefficienti
    salvati. NON ri-addestra nulla.
    Ritorna (clf, scaler):
      - clf: LogisticRegression con coef_ e intercept_ impostati manualmente
      - scaler: StandardScaler fittato sui dati di train (media/std)
    """
    case = f"{method}_{matcher}"
    sub  = df_coef[(df_coef["case"] == case) & (df_coef["model"] == "completo")]
    if len(sub) == 0:
        raise ValueError(f"Coefficienti mancanti per {case} / completo")

    coef_map = dict(zip(sub["feature"], sub["coef"]))

    # Ordine deve corrispondere a FEAT_COLS
    coef_vec  = np.array([[coef_map[f] for f in FEAT_COLS]])
    intercept = np.array([coef_map["intercept"]])

    clf = LogisticRegression()
    clf.coef_      = coef_vec
    clf.intercept_ = intercept
    clf.classes_   = np.array([0, 1])

    # StandardScaler: fit sul CSV train sun+night unito
    train_path = f"{FEAT_DIR}/features_{method}_{matcher}_sun_night.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File train non trovato: {train_path}")
    df_tr  = pd.read_csv(train_path)
    scaler = StandardScaler().fit(df_tr[FEAT_COLS].values)

    return clf, scaler

# ── STEP 2: funzione curva adattiva ───────────────────────────────────────────

def adaptive_curve(p_scores, retrieval_correct, reranked_correct):
    """
    Ordina per P(top1_correct) crescente (le query più incerte prima).
    Sweep: per ogni k query attivate usa reranked_correct, per le restanti
    retrieval_correct. Restituisce (frac_attivate, recall@1).
    """
    n      = len(p_scores)
    order  = np.argsort(p_scores)
    results = retrieval_correct.copy().astype(float)

    frac_list   = [0.0]
    recall_list = [float(results.mean())]

    for i, idx in enumerate(order):
        results[idx] = float(reranked_correct[idx])
        frac_list.append((i + 1) / n)
        recall_list.append(float(results.mean()))

    return np.array(frac_list), np.array(recall_list)

# ── STEP 3: calcola curve per tutte le configurazioni ─────────────────────────

print("\n=== Calcolo curve ===")
results = {}

for method in METHODS:
    results[method] = {}
    for matcher in MATCHERS:
        print(f"\n--- {method} × {matcher} ---")
        try:
            clf, scaler = load_logreg_from_coef(method, matcher)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        results[method][matcher] = {}

        for ds in TEST_DSS:
            path_te = f"{TEST_DIR}/features_{method}_{ds}_test_{matcher}.csv"
            if not os.path.exists(path_te):
                print(f"  SKIP {ds}: file non trovato")
                continue
            df_te = pd.read_csv(path_te)

            # Carica reranked_correct dal file debug
            debug_path = f"{TEST_DIR}/reranked_debug_{method}_{ds}_{matcher}.csv"
            if not os.path.exists(debug_path):
                print(f"  SKIP {ds}: reranked_debug non trovato")
                continue
            sub = pd.read_csv(debug_path)[["query_id", "reranked_correct"]]
            sub = sub[sub["reranked_correct"].notna()].copy()
            sub["reranked_correct"] = sub["reranked_correct"].astype(int)

            # Allineamento su query_id
            df_te["query_id_int"] = df_te["query_id"].astype(int)
            sub["query_id_int"]   = sub["query_id"].astype(int)
            merged = df_te[["query_id_int", LABEL_COL] + FEAT_COLS].merge(
                sub[["query_id_int", "reranked_correct"]],
                on="query_id_int", how="inner"
            )
            if len(merged) < len(df_te):
                print(f"  WARNING {ds}: {len(df_te)-len(merged)} query non allineate")

            X_m      = scaler.transform(merged[FEAT_COLS].values)
            p        = clf.predict_proba(X_m)[:, 1]
            ret_ok   = merged[LABEL_COL].values.astype(float)
            rerk_ok  = merged["reranked_correct"].values.astype(float)

            frac, rec = adaptive_curve(p, ret_ok, rerk_ok)

            r1_no   = float(ret_ok.mean())
            r1_full = float(rerk_ok.mean())

            # Breakeven: primo punto in cui la curva raggiunge il livello full
            be_idx  = int(np.searchsorted(rec, r1_full))
            be_frac = float(frac[be_idx]) if be_idx < len(frac) else 1.0

            # Soglia P(top1_correct) al breakeven
            order        = np.argsort(p)
            n_activated  = int(round(be_frac * len(p)))
            if n_activated > 0:
                be_threshold = float(p[order[min(n_activated - 1, len(p) - 1)]])
            else:
                be_threshold = float(p[order[0]])

            results[method][matcher][ds] = dict(
                frac=frac, rec=rec, p=p, order=order,
                r1_no=r1_no, r1_full=r1_full,
                be_frac=be_frac, be_threshold=be_threshold,
                n=len(merged), merged=merged
            )
            print(f"  {ds:9s}: retrieval={r1_no:.3f}  full={r1_full:.3f}  "
                  f"gain={r1_full-r1_no:+.3f}  breakeven={be_frac*100:.0f}%  "
                  f"saving={(1-be_frac)*100:.0f}%  "
                  f"threshold_P={be_threshold:.4f}")

# ── STEP 4: plot ──────────────────────────────────────────────────────────────

print("\n=== Plot ===")

n_rows = len(METHODS) * len(MATCHERS)
n_cols = len(TEST_DSS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4 * n_rows),
                         squeeze=False)

row = 0
for method in METHODS:
    for matcher in MATCHERS:
        for col, ds in enumerate(TEST_DSS):
            ax = axes[row][col]
            if matcher not in results.get(method, {}) or \
               ds not in results[method].get(matcher, {}):
                ax.set_visible(False)
                continue

            d = results[method][matcher][ds]
            frac, rec = d["frac"], d["rec"]

            ax.plot(frac * 100, rec * 100,
                    color="tab:blue", lw=2.5, label="Adaptive (regressor)")
            ax.axhline(d["r1_no"]   * 100, color="gray",    ls="--", lw=1.5,
                       label=f"No re-ranking ({d['r1_no']*100:.1f}%)")
            ax.axhline(d["r1_full"] * 100, color="tab:red",  ls="--", lw=1.5,
                       label=f"Full re-ranking ({d['r1_full']*100:.1f}%)")
            if d["be_frac"] < 1.0:
                ax.axvline(d["be_frac"] * 100, color="tab:green", ls=":", lw=1.5,
                           label=f"Breakeven {d['be_frac']*100:.0f}%  "
                                 f"(saving {(1-d['be_frac'])*100:.0f}%)")

            ax.set_title(f"{method} × {matcher}\n{ds}  (n={d['n']})", fontsize=9)
            ax.set_xlabel("% query re-ranked")
            ax.set_ylabel("Recall@1 (%)")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
        row += 1

plt.suptitle(
    "Adaptive Re-ranking: Recall@1 vs % query attivate\n"
    "(modello completo: clustered_score + relative_gap + d1 + num_inliers)",
    fontsize=11, y=1.01
)
plt.tight_layout()

out_pdf = f"{OUT_DIR}/adaptive_reranking_curve.pdf"
out_png = f"{OUT_DIR}/adaptive_reranking_curve.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSalvato:\n  {out_pdf}\n  {out_png}")

# ── STEP 5: tabella cost savings ──────────────────────────────────────────────

print("\n=== Tabella cost savings ===")
tab_rows = []
for method in METHODS:
    for matcher in MATCHERS:
        for ds in TEST_DSS:
            if matcher not in results.get(method, {}) or \
               ds not in results[method].get(matcher, {}):
                continue
            d = results[method][matcher][ds]
            tab_rows.append({
                "method":        method,
                "matcher":       matcher,
                "dataset":       ds,
                "R@1_retrieval": round(d["r1_no"]   * 100, 1),
                "R@1_full_rerk": round(d["r1_full"] * 100, 1),
                "gain":          round((d["r1_full"] - d["r1_no"]) * 100, 1),
                "breakeven_%":   round(d["be_frac"]  * 100, 1),
                "saving_%":      round((1 - d["be_frac"]) * 100, 1),
                "threshold_P":   round(d["be_threshold"], 4),
                "n_queries":     d["n"],
            })

df_tab = pd.DataFrame(tab_rows)
print(df_tab.to_string(index=False))
df_tab.to_csv(f"{OUT_DIR}/adaptive_reranking_savings.csv", index=False)
print(f"\nSalvato: {OUT_DIR}/adaptive_reranking_savings.csv")

# ── STEP 6: per-query scores ──────────────────────────────────────────────────

print("\n=== Salvo per-query scores ===")
pq_rows = []
for method in METHODS:
    for matcher in MATCHERS:
        for ds in TEST_DSS:
            if matcher not in results.get(method, {}) or \
               ds not in results[method].get(matcher, {}):
                continue
            d = results[method][matcher][ds]
            merged = d["merged"]
            p      = d["p"]
            for i, (_, row) in enumerate(merged.iterrows()):
                pq_rows.append({
                    "method":             method,
                    "matcher":            matcher,
                    "dataset":            ds,
                    "query_id":           int(row["query_id_int"]),
                    "p_top1_correct":     round(float(p[i]), 6),
                    "retrieval_correct":  int(row[LABEL_COL]),
                    "reranked_correct":   int(row["reranked_correct"]),
                    "activated":          int(float(p[i]) < d["be_threshold"]),
                })

df_pq = pd.DataFrame(pq_rows)
df_pq.to_csv(f"{OUT_DIR}/adaptive_reranking_perquery.csv", index=False)
print(f"Salvato: {OUT_DIR}/adaptive_reranking_perquery.csv  ({len(df_pq)} righe)")

# ── STEP 7: punti curva + AUC ─────────────────────────────────────────────────

print("\n=== Salvo punti curva e AUC ===")
curve_rows = []
for method in METHODS:
    for matcher in MATCHERS:
        for ds in TEST_DSS:
            if matcher not in results.get(method, {}) or \
               ds not in results[method].get(matcher, {}):
                continue
            d    = results[method][matcher][ds]
            frac = d["frac"]
            rec  = d["rec"]
            auc  = float(np.trapz(rec, frac))   # area sotto la curva adattiva
            for f, r in zip(frac, rec):
                curve_rows.append({
                    "method":  method,
                    "matcher": matcher,
                    "dataset": ds,
                    "frac_activated": round(float(f), 6),
                    "recall_at_1":    round(float(r), 6),
                    "auc":            round(auc, 6),
                })

df_curve = pd.DataFrame(curve_rows)
df_curve.to_csv(f"{OUT_DIR}/adaptive_reranking_curve_points.csv", index=False)
print(f"Salvato: {OUT_DIR}/adaptive_reranking_curve_points.csv")

# Stampa AUC per configurazione
print("\nAUC curva adattiva (area sotto Recall@1 vs % attivate):")
auc_summary = (df_curve.groupby(["method","matcher","dataset"])["auc"]
               .first().unstack("dataset"))
print(auc_summary.to_string())
