#!/usr/bin/env python3
"""
Estrai i tempi di retrieval da tutti i z_data.torch
Esegui su Google Colab dopo aver montato Drive.

Struttura attesa:
  logs/retrieval/{method}/{dataset}/default_metric/{timestamp}/z_data.torch
"""

import os
import glob
import torch
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE = "/content/drive/MyDrive/Progetto_Machine_Learning/logs/retrieval"

METHODS = [
    "cosplace_resnet18_512",
    "megaloc_resnet50_4096",
    "mixvpr_resnet50_4096",
    "netvlad_vgg16_4096",
]

# Dataset di test (esclude train e val)
TEST_DATASETS = ["svox_sun", "svox_night", "sf_xs", "tokyo_xs"]

# ── FUNZIONI ──────────────────────────────────────────────────────────────────

def find_latest_zdata(method_dir: str, dataset: str) -> str | None:
    """
    Trova il z_data.torch più recente per un dato metodo/dataset.
    Esclude cartelle *_train e *_val.
    """
    pattern = os.path.join(method_dir, dataset, "default_metric", "*", "z_data.torch")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]  # più recente = ultimo alfabeticamente (timestamp)


def extract_times(path: str) -> dict:
    """Carica z_data.torch ed estrae i tempi."""
    data = torch.load(path, weights_only=False)

    n_queries = data["predictions"].shape[0]

    if "time_extraction" in data and "time_retrieval" in data:
        t_ext = float(data["time_extraction"])
        t_ret = float(data["time_retrieval"])
        t_tot = t_ext + t_ret
        return {
            "n_queries":     n_queries,
            "t_extraction_s": round(t_ext, 4),
            "t_retrieval_s":  round(t_ret, 4),
            "t_total_s":      round(t_tot, 4),
            "ms_per_query":   round(t_tot / n_queries * 1000, 3),
        }
    else:
        return {
            "n_queries":     n_queries,
            "t_extraction_s": None,
            "t_retrieval_s":  None,
            "t_total_s":      None,
            "ms_per_query":   None,
        }


# ── MAIN ──────────────────────────────────────────────────────────────────────

rows = []

for method in METHODS:
    method_dir = os.path.join(BASE, method)
    if not os.path.isdir(method_dir):
        print(f"[WARN] Cartella non trovata: {method_dir}")
        continue

    for ds in TEST_DATASETS:
        path = find_latest_zdata(method_dir, ds)
        if path is None:
            print(f"[WARN] Nessun z_data.torch per {method} / {ds}")
            rows.append({"method": method, "dataset": ds,
                         "n_queries": None, "t_extraction_s": None,
                         "t_retrieval_s": None, "t_total_s": None,
                         "ms_per_query": None})
            continue

        print(f"Carico: {path}")
        try:
            times = extract_times(path)
            rows.append({"method": method, "dataset": ds, **times})
            print(f"  n_queries={times['n_queries']}  "
                  f"t_ext={times['t_extraction_s']}s  "
                  f"t_ret={times['t_retrieval_s']}s  "
                  f"t_tot={times['t_total_s']}s  "
                  f"({times['ms_per_query']} ms/query)")
        except Exception as e:
            print(f"  [ERROR] {e}")
            rows.append({"method": method, "dataset": ds,
                         "n_queries": None, "t_extraction_s": None,
                         "t_retrieval_s": None, "t_total_s": None,
                         "ms_per_query": None})

# ── TABELLA PIVOT: tempo totale (s) ──────────────────────────────────────────
df = pd.DataFrame(rows)

print("\n" + "="*70)
print("TABELLA: Tempo TOTALE (estrazione + retrieval) [secondi]")
print("="*70)
pivot_tot = df.pivot(index="method", columns="dataset", values="t_total_s")
pivot_tot = pivot_tot[TEST_DATASETS]  # ordine colonne
print(pivot_tot.to_string())

print("\n" + "="*70)
print("TABELLA: Tempo per singola query [ms/query]")
print("="*70)
pivot_ms = df.pivot(index="method", columns="dataset", values="ms_per_query")
pivot_ms = pivot_ms[TEST_DATASETS]
print(pivot_ms.to_string())

print("\n" + "="*70)
print("TABELLA: Tempo ESTRAZIONE descrittori [secondi]")
print("="*70)
pivot_ext = df.pivot(index="method", columns="dataset", values="t_extraction_s")
pivot_ext = pivot_ext[TEST_DATASETS]
print(pivot_ext.to_string())

print("\n" + "="*70)
print("TABELLA: Tempo RETRIEVAL FAISS [secondi]")
print("="*70)
pivot_ret = df.pivot(index="method", columns="dataset", values="t_retrieval_s")
pivot_ret = pivot_ret[TEST_DATASETS]
print(pivot_ret.to_string())

print("\n" + "="*70)
print("TABELLA COMPLETA (raw)")
print("="*70)
print(df.to_string(index=False))

# Salva CSV
out_csv = "/content/drive/MyDrive/Progetto_Machine_Learning/retrieval_times.csv"
df.to_csv(out_csv, index=False)
print(f"\nSalvato: {out_csv}")
