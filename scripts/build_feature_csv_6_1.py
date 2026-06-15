#!/usr/bin/env python3
# ============================================================================
#  Estensione 6.1 VPR  -  Costruzione tabella feature (un CSV per caso)
#  Da lanciare in Colab, con il Drive gia' montato (zero download).
#
#  Per ciascun caso  metodo x dominio  produce un CSV con UNA RIGA PER QUERY:
#     d1, d2, gap, relative_gap, num_inliers, inlier_ratio,
#     clustered_score, top1_correct           (+ query_id, num_matches)
#
#  - d1, d2            -> distanze descrittore dal retrieval (z_data.torch)
#  - top1_correct      -> 1 se la top-1 e' entro 25 m (distanza GPS dai filename)
#  - num_inliers       -> dai .torch del matching (un file per query)
#  - inlier_ratio      -> num_inliers / num_matches
#  - clustered_score   -> DBSCAN sulle coordinate (x,y) degli inlier sulla query,
#                         normalizzate in [0,1], clustered_score = sum_c sqrt(n_c)
# ============================================================================

import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------
# 1) CONFIG  -  adatta solo questi valori
# ----------------------------------------------------------------------------
RETRIEVAL_ROOT = "/content/drive/MyDrive/Progetto_Machine_Learning/logs/retrieval"
OUT_DIR        = "/content/drive/MyDrive/Progetto_Machine_Learning/feature_csv_6_1"

MATCHER = "superpoint-lg"          # cartella matching = preds_<MATCHER>

# I 4 casi: (nome_metodo_cartella, nome_dataset_cartella)
CASES = [
    ("megaloc_resnet50_4096", "sf_xs_val"),
    ("netvlad_vgg16_4096",    "sf_xs_val")
]

# Parametri fissi (valori "classici" da letteratura, NON ottimizzati sul test)
POS_DIST_THRESHOLD = 25      # metri: top-1 corretta se entro 25 m
IMG_SIZE           = 512     # le coord del matching sono in 512x512 (--im-size default)
DBSCAN_EPS         = 0.1     # in spazio normalizzato [0,1]
DBSCAN_MIN_SAMPLES = 4

# ----------------------------------------------------------------------------
# 2) Helper (riprodotti da util.py per non dipendere dal cwd)
# ----------------------------------------------------------------------------
def read_file_preds(preds_txt_file):
    with open(preds_txt_file) as f:
        lines = f.read().splitlines()
    query_path = lines[1]
    preds_paths = lines[4:lines.index('', 4)]
    return query_path, preds_paths

def get_utm_from_path(path):
    return np.array([path.split("@")[1], path.split("@")[2]]).astype(np.float32)

def geo_dist(a, b):
    return float(((a - b) ** 2).sum() ** 0.5)

def top1_geo_distance(txt_file):
    """Distanza GPS (m) tra la query e la sua predizione top-1."""
    q_path, pred_paths = read_file_preds(txt_file)
    q_utm  = get_utm_from_path(q_path)
    p1_utm = get_utm_from_path(pred_paths[0])
    return geo_dist(q_utm, p1_utm)

def clustered_score(inlier_kpts0, img_size):
    """DBSCAN sulle coord inlier sulla query, normalizzate -> sum_c sqrt(n_c)."""
    if inlier_kpts0 is None:
        return 0.0
    pts = np.asarray(inlier_kpts0, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    pts_norm = pts / float(img_size)                      # step 1 del PDF: in [0,1]
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(pts_norm)
    score = 0.0
    for lab in set(labels):
        if lab == -1:                                     # -1 = noise, escluso
            continue
        n_c = int((labels == lab).sum())
        score += math.sqrt(n_c)                           # contributo sublineare
    return float(score)

# ----------------------------------------------------------------------------
# 3) Localizza i percorsi di un caso (auto-discovery del timestamp)
# ----------------------------------------------------------------------------
def find_case_paths(method_dir, dataset_dir):
    base = os.path.join(RETRIEVAL_ROOT, method_dir, dataset_dir, "default_metric")
    runs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    if not runs:
        raise FileNotFoundError(f"Nessun run trovato in {base}")
    run = sorted(runs)[-1]                                # il piu' recente
    z_data    = os.path.join(run, "z_data.torch")
    preds_dir = os.path.join(run, "preds")
    match_dir = os.path.join(run, f"preds_{MATCHER}")
    return run, z_data, preds_dir, match_dir

# ----------------------------------------------------------------------------
# 4) Costruisci la tabella di un caso
# ----------------------------------------------------------------------------
def build_table(method_dir, dataset_dir):
    run, z_data_path, preds_dir, match_dir = find_case_paths(method_dir, dataset_dir)
    print(f"\n=== {method_dir} | {dataset_dir} ===")
    print(f"    run:      {run}")
    for p in (z_data_path, preds_dir, match_dir):
        print(f"    {'OK ' if os.path.exists(p) else 'MANCA'} {p}")

    # --- retrieval: d1, d2 (distanze descrittore) ---
    z = torch.load(z_data_path, weights_only=False)
    dists = z["distances"]                                # dists[i][0]=d1, [1]=d2

    # --- query ordinate come in main.py / eval.py ---
    txt_files = glob.glob(os.path.join(preds_dir, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    rows = []
    n_missing_match = 0
    for itr, txt_file in enumerate(txt_files):
        stem = Path(txt_file).stem                        # nome query (es. "23")
        d1 = float(dists[itr][0])
        d2 = float(dists[itr][1])
        gap = d2 - d1
        relative_gap = gap / d1 if d1 != 0 else 0.0

        # label dalla ground-truth GPS
        top1_correct = 1 if top1_geo_distance(txt_file) <= POS_DIST_THRESHOLD else 0

        # matching: un .torch per query (lista len=num_preds, prendiamo [0])
        match_path = os.path.join(match_dir, f"{stem}.torch")
        if os.path.exists(match_path):
            m = torch.load(match_path, weights_only=False)[0]
            num_inliers = int(m["num_inliers"])
            mk0 = m.get("matched_kpts0")
            num_matches = int(len(mk0)) if mk0 is not None else 0
            inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
            cscore = clustered_score(m.get("inlier_kpts0"), IMG_SIZE)
        else:
            n_missing_match += 1
            num_inliers, num_matches, inlier_ratio, cscore = np.nan, np.nan, np.nan, np.nan

        rows.append(dict(
            query_id=stem, d1=d1, d2=d2, gap=gap, relative_gap=relative_gap,
            num_inliers=num_inliers, num_matches=num_matches,
            inlier_ratio=inlier_ratio, clustered_score=cscore,
            top1_correct=top1_correct,
        ))

    df = pd.DataFrame(rows, columns=[
        "query_id", "d1", "d2", "gap", "relative_gap",
        "num_inliers", "num_matches", "inlier_ratio",
        "clustered_score", "top1_correct",
    ])
    print(f"    query: {len(df)} | match mancanti: {n_missing_match} "
          f"| top1_correct medio: {df['top1_correct'].mean():.3f}")
    return df

# ----------------------------------------------------------------------------
# 5) Run su tutti i casi
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for method_dir, dataset_dir in CASES:
        try:
            df = build_table(method_dir, dataset_dir)
        except FileNotFoundError as e:
            print(f"    SALTATO: {e}")
            continue
        method = method_dir.split("_")[0]                 # megaloc / netvlad
        dom = "sun" if "sun" in dataset_dir else "sf_xs_val"
        out = os.path.join(OUT_DIR, f"features_{method}_{dom}_{MATCHER}.csv")
        df.to_csv(out, index=False)
        print(f"    -> salvato {out}")

if __name__ == "__main__":
    main()
