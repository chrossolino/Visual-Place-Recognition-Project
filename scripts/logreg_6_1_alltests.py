#!/usr/bin/env python3
"""
Estensione 6.1 VPR - Regressione logistica, split del professore, TUTTI i test.
  TRAIN = 100% SVOX train (sun+night)         features_<m>_<mt>_sun_night.csv
  VAL   = SF-XS val (iperparametri)           features_<m>_sf_xs_val_<mt>.csv
  TEST  = svox_sun, svox_night, sf_xs, tokyo_xs  (4 dataset separati)
Modello 'completo' ORA include num_inliers:
  completo = clustered_score + relative_gap + d1 + num_inliers
Per ogni (metodo x matcher) si tara C e soglia su VAL, si congela, e si valuta
su ciascun test set -> 16 pannelli ROC (metodo x matcher x dataset).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

DATA = "/content/drive/MyDrive/Progetto_Machine_Learning/Visual-Place-Recognition-Project"
TESTD = os.path.join(DATA, "feature_csv_6_1", "features_test")
OUTDIR = os.path.join(DATA, "results_6_1_alltests")
os.makedirs(OUTDIR, exist_ok=True)

METHODS = ["megaloc", "netvlad"]
MATCHERS = ["superpoint-lg", "superglue"]
# dataset label -> token nel filename del test
DATASETS = {"svox_sun": "sun", "svox_night": "night", "sf_xs": "sf_xs", "tokyo_xs": "tokyo_xs"}

ALL_FEATURES = ["d1", "d2", "gap", "relative_gap", "num_inliers", "inlier_ratio", "clustered_score"]
MODELS = {
    "baseline_gap":         ["gap"],
    "baseline_num_inliers": ["num_inliers"],
    "proposta_clustered":   ["clustered_score"],
    "completo":             ["clustered_score", "relative_gap", "d1", "num_inliers"],
}
TARGET = "top1_correct"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

def metrics(y, pred, proba):
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
    )

def best_threshold(y, score):
    y = np.asarray(y); score = np.asarray(score)
    cand = np.unique(np.quantile(score, np.linspace(0, 1, 301)))
    P = y.sum(); bt, bf = 0.5, -1.0
    for t in cand:
        pred = score >= t
        tp = np.logical_and(pred, y == 1).sum(); pp = pred.sum()
        prec = tp / pp if pp > 0 else 0.0
        rec = tp / P if P > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f > bf:
            bf, bt = f, t
    return float(bt), float(bf)

rows_models, rows_coef = [], []
# roc_data[(method,matcher)][dataset][model] = (fpr,tpr,auc)
roc_data = {}
missing = []

for method in METHODS:
    for matcher in MATCHERS:
        key = f"{method}_{matcher}"
        ftr = os.path.join(DATA, "feature_csv_6_1", "features_train", f"features_{method}_{matcher}_sun_night.csv")
        fva = os.path.join(DATA, "feature_csv_6_1", "features_val", f"features_{method}_sf_xs_val_{matcher}.csv")
        if not (os.path.exists(ftr) and os.path.exists(fva)):
            print(f"!! manca train/val per {key}"); continue
        dtr = pd.read_csv(ftr); dva = pd.read_csv(fva)
        Xtr, ytr = dtr[ALL_FEATURES].values, dtr[TARGET].astype(int).values
        Xva, yva = dva[ALL_FEATURES].values, dva[TARGET].astype(int).values
        scaler = StandardScaler().fit(Xtr)
        Str, Sva = scaler.transform(Xtr), scaler.transform(Xva)
        idx = {f: i for i, f in enumerate(ALL_FEATURES)}

        # addestra ogni modello (C e soglia su VAL) -> frozen
        frozen = {}
        for mname, feats in MODELS.items():
            cols = [idx[f] for f in feats]
            bestC, bestAuc = 1.0, -1.0
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=1000).fit(Str[:, cols], ytr)
                va = roc_auc_score(yva, clf.predict_proba(Sva[:, cols])[:, 1]) if len(set(yva)) > 1 else np.nan
                if not np.isnan(va) and va > bestAuc:
                    bestAuc, bestC = va, C
            clf = LogisticRegression(C=bestC, max_iter=1000).fit(Str[:, cols], ytr)
            thr, f1v = best_threshold(yva, clf.predict_proba(Sva[:, cols])[:, 1])
            frozen[mname] = (clf, cols, bestC, bestAuc, thr, f1v)
            for f, c in zip(feats, clf.coef_[0]):
                rows_coef.append(dict(case=key, model=mname, feature=f, coef=float(c)))
            rows_coef.append(dict(case=key, model=mname, feature="intercept", coef=float(clf.intercept_[0])))

        roc_data[key] = {}
        for dlabel, tok in DATASETS.items():
            fte = os.path.join(TESTD, f"features_{method}_{tok}_test_{matcher}.csv")
            if not os.path.exists(fte):
                missing.append(f"{key} | {dlabel}")
                roc_data[key][dlabel] = None
                continue
            dte = pd.read_csv(fte)
            Xte, yte = dte[ALL_FEATURES].values, dte[TARGET].astype(int).values
            Ste = scaler.transform(Xte)
            roc_data[key][dlabel] = {}
            for mname, (clf, cols, bestC, bestAuc, thr, f1v) in frozen.items():
                proba = clf.predict_proba(Ste[:, cols])[:, 1]
                pred = (proba >= thr).astype(int)
                m = metrics(yte, pred, proba)
                rows_models.append(dict(case=key, method=method, matcher=matcher, dataset=dlabel,
                                        model=mname, C=bestC, val_auc=round(bestAuc, 4),
                                        val_threshold=round(thr, 4), val_f1=round(f1v, 4),
                                        n_test=len(yte), pos_rate=round(yte.mean(), 3), **m))
                fpr, tpr, _ = roc_curve(yte, proba)
                roc_data[key][dlabel][mname] = (fpr, tpr, m["roc_auc"])

res = pd.DataFrame(rows_models)
res.to_csv(os.path.join(OUTDIR, "metrics_models_test.csv"), index=False)
pd.DataFrame(rows_coef).to_csv(os.path.join(OUTDIR, "coefficients.csv"), index=False)

# ---- ROC: PDF con 4 pannelli per pagina (una pagina per metodo x matcher) ----
from matplotlib.backends.backend_pdf import PdfPages
keys = [f"{m}_{mt}" for m in METHODS for mt in MATCHERS]
dlabels = list(DATASETS.keys())
pdf_path = os.path.join(OUTDIR, "roc_per_pagina_4.pdf")
with PdfPages(pdf_path) as pdf:
    for key in keys:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))   # 4 pannelli/pagina, A4-ish
        for ax, dlabel in zip(axes.ravel(), dlabels):
            cell = roc_data.get(key, {}).get(dlabel)
            if cell:
                for mname, (fpr, tpr, auc) in cell.items():
                    ax.plot(fpr, tpr, lw=1.6, label=f"{mname} (AUC={auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.legend(fontsize=8, loc="lower right")
            else:
                ax.text(0.5, 0.5, "N/D", ha="center", va="center", fontsize=16, color="red")
            ax.set_title(dlabel, fontsize=12)
            ax.set_xlabel("FPR", fontsize=10); ax.set_ylabel("TPR", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        fig.suptitle(f"ROC sul test - {key}\n(train=SVOX, val=SF-XS, completo=clustered+rel_gap+d1+num_inliers)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=200)
        plt.close(fig)
        # anche PNG ad alta risoluzione per ogni pagina (utile da inserire singolarmente)
        fig.savefig(os.path.join(OUTDIR, f"roc_{key}.png"), dpi=200)
print("PDF ROC (4/pagina):", pdf_path)

pd.set_option("display.width", 250, "display.max_columns", 40)
print("\n=== ROC-AUC per (caso, dataset, modello) ===")
piv = res.pivot_table(index=["case", "dataset"], columns="model", values="roc_auc")
print(piv.round(3).to_string())

print("\n=== Soglia val (F1-ottimizzata su SF-XS val) per modello 'completo' ===")
thr_tab = (res[res["model"] == "completo"][["case", "val_threshold", "val_f1"]]
           .drop_duplicates("case")
           .set_index("case"))
print(thr_tab.round(4).to_string())

print("\nPannelli mancanti:", missing if missing else "nessuno")
print("Output in:", OUTDIR)
