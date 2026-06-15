#!/usr/bin/env python3
# ============================================================
# Estensione 6.1 VPR — R@1 post-reranking da file .torch
# Loop automatico su tutti i casi (method × dataset × matcher)
# ============================================================

import sys, shutil, time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path("/content/drive/MyDrive/Progetto_Machine_Learning/Visual-Place-Recognition-Project")
RETRIEVAL_ROOT = Path("/content/drive/MyDrive/Progetto_Machine_Learning/logs/retrieval")
OUT_DIR        = Path("/content/drive/MyDrive/Progetto_Machine_Learning/feature_csv_6_1/test")
CONSOLIDATO    = OUT_DIR / "rerank_eval_consolidato.csv"

# Copia i .torch su disco locale prima di leggerli (10-50x più veloce che da Drive)
LOCAL_CACHE = Path("/content/torch_cache")

POSITIVE_DIST_THRESHOLD = 25   # metri
NUM_PREDS               = 20   # candidati per query

# (method_dir, dataset_dir, ds_label)
CASES = [
    ("megaloc_resnet50_4096", "svox_sun",   "sun"),
    ("megaloc_resnet50_4096", "svox_night", "night"),
    ("megaloc_resnet50_4096", "sf_xs",      "sf_xs"),
    ("megaloc_resnet50_4096", "tokyo_xs",   "tokyo_xs"),
    ("netvlad_vgg16_4096",   "svox_sun",   "sun"),
    ("netvlad_vgg16_4096",   "svox_night", "night"),
    ("netvlad_vgg16_4096",   "sf_xs",      "sf_xs"),
    ("netvlad_vgg16_4096",   "tokyo_xs",   "tokyo_xs"),
]
MATCHERS = ["superglue", "superpoint-lg"]

# ── SETUP ─────────────────────────────────────────────────────────────────────
sys.path.append(str(PROJECT_DIR))
from util import get_list_distances_from_preds

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HELPER ────────────────────────────────────────────────────────────────────
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def extract_num_inliers(item):
    value = item.get("num_inliers", None)
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return int(value)

def find_run(method_dir, dataset_dir):
    base = RETRIEVAL_ROOT / method_dir / dataset_dir / "default_metric"
    runs = sorted([d for d in base.glob("*") if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"Nessun run in {base}")
    return runs[-1]

def cache_dir(src_dir: Path, label: str) -> Path:
    """Copia src_dir su disco locale se non già presente. Ritorna il path locale."""
    dst = LOCAL_CACHE / label
    if dst.exists():
        print(f"  cache già presente: {dst}")
        return dst
    print(f"  copia {src_dir.name} → {dst} ...", end=" ", flush=True)
    t0 = time.time()
    shutil.copytree(src_dir, dst)
    print(f"fatto in {time.time()-t0:.0f}s  ({sum(1 for _ in dst.glob('*'))} file)")
    return dst

# ── CORE ──────────────────────────────────────────────────────────────────────
def process_case(method_dir, dataset_dir, ds_label, matcher):
    method = method_dir.split("_")[0]
    run    = find_run(method_dir, dataset_dir)

    matching_dir_drive = run / f"preds_{matcher}"
    pred_txt_dir       = run / "preds"   # i .txt sono piccoli, si leggono da Drive

    if not matching_dir_drive.exists():
        print(f"  MANCA: {matching_dir_drive}")
        return None

    # Copia .torch su disco locale
    cache_label  = f"{method}_{ds_label}_{matcher}"
    matching_dir = cache_dir(matching_dir_drive, cache_label)

    torch_files = sorted(
        [f for f in matching_dir.glob("*.torch") if f.stem.isdigit()],
        key=lambda x: int(x.stem)
    )
    if not torch_files:
        print(f"  Nessun .torch in {matching_dir}")
        return None

    rows = []
    for torch_file in torch_files:
        query_stem   = torch_file.stem
        query_id     = int(query_stem)
        pred_txt_file = pred_txt_dir / f"{query_stem}.txt"

        row = {
            "query_id":                  query_id,
            "best_rank_originale_0based": None,
            "best_num_inliers":           None,
            "top1_distance_m":            np.nan,
            "reranked_correct":           np.nan,
            "error":                      None,
        }

        try:
            data = safe_torch_load(torch_file)
            if not isinstance(data, list):
                row["error"] = "non è una lista"
                rows.append(row)
                continue

            inliers = []
            for item in data[:NUM_PREDS]:
                inliers.append(extract_num_inliers(item) if isinstance(item, dict) else None)

            valid_pos = [i for i, x in enumerate(inliers) if x is not None]
            if not valid_pos:
                row["error"] = "nessun num_inliers valido"
                rows.append(row)
                continue

            best_rank        = max(valid_pos, key=lambda i: inliers[i])
            row["best_rank_originale_0based"] = best_rank
            row["best_num_inliers"]           = inliers[best_rank]

            if not pred_txt_file.exists():
                row["error"] = "txt mancante"
                rows.append(row)
                continue

            distances = np.asarray(get_list_distances_from_preds(pred_txt_file), dtype=float)
            dist      = float(distances[best_rank])
            row["top1_distance_m"]  = dist
            row["reranked_correct"] = 1 if dist <= POSITIVE_DIST_THRESHOLD else 0

        except Exception as e:
            row["error"] = repr(e)

        rows.append(row)

    df = pd.DataFrame(rows)
    valid = df["reranked_correct"].dropna()
    r1    = valid.mean() if len(valid) > 0 else float("nan")
    n_err = df["error"].notna().sum()
    print(f"  OK {method:8s} {ds_label:9s} {matcher:13s} → "
          f"{len(df):4d} query  R@1={r1:.3f}  errori={n_err}")

    return df, method, matcher, ds_label

# ── LOOP ──────────────────────────────────────────────────────────────────────
print("=== Build reranked_correct da .torch ===\n")

all_clean_rows = []

for method_dir, dataset_dir, ds_label in CASES:
    for matcher in MATCHERS:
        print(f"--- {method_dir.split('_')[0]} × {matcher} × {ds_label} ---")
        try:
            result = process_case(method_dir, dataset_dir, ds_label, matcher)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        if result is None:
            continue

        df, method, matcher_out, ds = result

        # CSV debug completo
        debug_path = OUT_DIR / f"reranked_debug_{method}_{ds}_{matcher_out}.csv"
        if not debug_path.exists():
            df.to_csv(debug_path, index=False)
            print(f"  → debug: {debug_path.name}")
        else:
            print(f"  (debug già esistente, non sovrascritto)")

        # Righe per il consolidato (solo query senza errori)
        for _, row in df[df["error"].isna()].iterrows():
            all_clean_rows.append({
                "query_id":         int(row["query_id"]),
                "method":           method,
                "matcher":          matcher_out,
                "dataset":          ds,
                "reranked_correct": int(row["reranked_correct"]),
            })

# ── AGGIORNA CONSOLIDATO ──────────────────────────────────────────────────────
print("\n=== Aggiorno consolidato ===")
df_new = pd.DataFrame(all_clean_rows)

if CONSOLIDATO.exists():
    df_old = pd.read_csv(CONSOLIDATO)
    key_cols = ["method", "matcher", "dataset"]
    new_keys = df_new[key_cols].drop_duplicates()
    mask = pd.merge(df_old[key_cols], new_keys, on=key_cols,
                    how="left", indicator=True)["_merge"] == "left_only"
    df_merged = pd.concat([df_old[mask], df_new], ignore_index=True)
    print(f"Righe precedenti mantenute: {mask.sum()}")
else:
    df_merged = df_new

df_merged.to_csv(CONSOLIDATO, index=False)
print(f"Salvato: {CONSOLIDATO}  ({len(df_merged)} righe totali)\n")
print(df_merged.groupby(["method", "matcher", "dataset"])["reranked_correct"]
      .mean().unstack().to_string())
