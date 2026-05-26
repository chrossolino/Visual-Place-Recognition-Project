import os
import csv
import json
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from util import get_list_distances_from_preds


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with predictions of a VPR model",
    )

    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="Directory with image matching results already computed on the top candidates",
    )

    parser.add_argument(
        "--z-data",
        type=str,
        required=True,
        help="Path to z_data.torch saved by main.py with --save_for_uncertainty",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory where adaptive re-ranking results will be saved",
    )

    parser.add_argument(
        "--max-num-preds",
        type=int,
        default=20,
        help="Maximum number of predictions available for adaptive re-ranking",
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Adaptive K values: small, medium, large",
    )

    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance in meters for a prediction to be considered positive",
    )

    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Recall values to compute",
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=35.0,
        help="Lower percentile for the uncertainty gap",
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=70.0,
        help="Higher percentile for the uncertainty gap",
    )

    return parser.parse_args()


def choose_adaptive_k(gap, low_threshold, high_threshold, k_values):
    """
    Choose the number of candidates to re-rank according to retrieval uncertainty.

    In this pipeline, FAISS uses L2 distance:
    - lower distance means better candidate;
    - gap = distance_2 - distance_1.

    Large gap  -> confident query    -> smaller K.
    Medium gap -> medium uncertainty -> intermediate K.
    Small gap  -> uncertain query    -> larger K.
    """

    k_small, k_medium, k_large = k_values

    if gap >= high_threshold:
        return k_small
    elif gap >= low_threshold:
        return k_medium
    else:
        return k_large


def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    z_data_path = Path(args.z_data)

    if args.out_dir is None:
        output_folder = inliers_folder.parent / f"adaptive_reranking_results_{inliers_folder.name}"
    else:
        output_folder = Path(args.out_dir)

    output_folder.mkdir(parents=True, exist_ok=True)

    max_num_preds = args.max_num_preds
    k_values = args.k_values
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    if len(k_values) != 3:
        raise ValueError("--k-values must contain exactly three values, for example: 5 10 20")

    if max(recall_values) > max_num_preds:
        raise ValueError("The maximum recall value cannot be larger than --max-num-preds")

    if max(k_values) > max_num_preds:
        raise ValueError("The maximum adaptive K cannot be larger than --max-num-preds")

    # Load global retrieval data saved by main.py with --save_for_uncertainty.
    z_data = torch.load(z_data_path, weights_only=False)
    faiss_distances = z_data["distances"]

    if faiss_distances.shape[1] < max_num_preds:
        raise ValueError(
            f"z_data contains only {faiss_distances.shape[1]} predictions, "
            f"but --max-num-preds is {max_num_preds}."
        )

    # Uncertainty measure: gap between the first and second FAISS distances.
    gaps = faiss_distances[:, 1] - faiss_distances[:, 0]

    low_threshold = np.percentile(gaps, args.low_percentile)
    high_threshold = np.percentile(gaps, args.high_percentile)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)

    if total_queries != len(gaps):
        raise ValueError(
            f"Number of txt prediction files ({total_queries}) is different from "
            f"number of queries in z_data ({len(gaps)}). "
            f"Check that preds-dir and z-data belong to the same run."
        )

    recalls = np.zeros(len(recall_values))
    used_ks = []
    per_query_results = []

    for query_idx, txt_file_query in enumerate(tqdm(txt_files, desc="Adaptive re-ranking")):
        query_name = Path(txt_file_query).stem

        # Geographic distances are used only for evaluation.
        original_geo_dists = torch.tensor(
            get_list_distances_from_preds(txt_file_query),
            dtype=torch.float32
        )[:max_num_preds]

        gap = gaps[query_idx]
        adaptive_k = choose_adaptive_k(gap, low_threshold, high_threshold, k_values)
        used_ks.append(adaptive_k)

        torch_file_query = inliers_folder / f"{query_name}.torch"

        if not torch_file_query.exists():
            raise FileNotFoundError(
                f"Missing matching file: {torch_file_query}\n"
                f"Check if --inliers-dir is correct."
            )

        query_results = torch.load(torch_file_query, weights_only=False)

        if len(query_results) < adaptive_k:
            raise ValueError(
                f"{torch_file_query} contains only {len(query_results)} matching results, "
                f"but adaptive K is {adaptive_k}."
            )

        # Local score used for re-ranking: number of geometric inliers.
        query_db_inliers = torch.zeros(adaptive_k, dtype=torch.float32)

        for i in range(adaptive_k):
            query_db_inliers[i] = query_results[i]["num_inliers"]

        _, reranked_indices = torch.sort(query_db_inliers, descending=True)

        # Only the first adaptive_k candidates are re-ranked.
        # The remaining candidates are appended in their original global retrieval order.
        # This keeps Recall@10 and Recall@20 meaningful even when adaptive_k is 5 or 10.
        remaining_indices = torch.arange(adaptive_k, max_num_preds)
        final_indices = torch.cat([reranked_indices, remaining_indices])

        final_geo_dists = original_geo_dists[final_indices]

        # Per-query recall indicators.
        query_recall = {}
        for n in recall_values:
            query_recall[f"R@{n}"] = int(torch.any(final_geo_dists[:n] <= threshold).item())

        per_query_results.append(
            {
                "query": query_name,
                "gap": float(gap),
                "adaptive_k": int(adaptive_k),
                "top1_geo_dist_after_reranking": float(final_geo_dists[0]),
                "top1_original_position_after_reranking": int(final_indices[0]),
                **query_recall,
            }
        )

        # Save query-level adaptive re-ranking output.
        torch.save(
            {
                "query": query_name,
                "gap": float(gap),
                "adaptive_k": int(adaptive_k),
                "local_inliers_used_for_reranking": query_db_inliers,
                "reranked_indices_within_adaptive_k": reranked_indices,
                "final_indices": final_indices,
                "original_geo_dists": original_geo_dists,
                "final_geo_dists": final_geo_dists,
            },
            output_folder / f"{query_name}.torch",
        )

        # Dataset-level cumulative recall.
        for i, n in enumerate(recall_values):
            if torch.any(final_geo_dists[:n] <= threshold):
                recalls[i:] += 1
                break

    recalls = recalls / total_queries * 100

    used_ks = np.array(used_ks)
    avg_k = used_ks.mean()

    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)]
    )

    print("\nAdaptive Re-ranking results")
    print("---------------------------")
    print(recalls_str)

    print("\nAdaptive K statistics")
    print("---------------------")
    print(f"low threshold  percentile {args.low_percentile:.1f}:  {low_threshold:.6f}")
    print(f"high threshold percentile {args.high_percentile:.1f}: {high_threshold:.6f}")
    print(f"Average K: {avg_k:.2f}")

    for k in k_values:
        percentage = np.mean(used_ks == k) * 100
        print(f"Queries with K={k}: {percentage:.1f}%")

    reduction = (1 - avg_k / max_num_preds) * 100

    print(
        f"\nAverage local matching reduction compared to fixed Top-{max_num_preds}: "
        f"{reduction:.1f}%"
    )

    # Save summary as JSON.
    summary = {
        "recalls": {f"R@{val}": float(rec) for val, rec in zip(recall_values, recalls)},
        "low_percentile": float(args.low_percentile),
        "high_percentile": float(args.high_percentile),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "average_k": float(avg_k),
        "k_values": [int(k) for k in k_values],
        "k_distribution_percent": {
            str(int(k)): float(np.mean(used_ks == k) * 100)
            for k in k_values
        },
        "average_local_matching_reduction_percent": float(reduction),
        "max_num_preds": int(max_num_preds),
        "positive_dist_threshold": int(threshold),
        "total_queries": int(total_queries),
        "preds_dir": str(preds_folder),
        "inliers_dir": str(inliers_folder),
        "z_data": str(z_data_path),
        "output_folder": str(output_folder),
    }

    with open(output_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Save per-query results as CSV.
    with open(output_folder / "per_query_results.csv", "w", newline="") as f:
        fieldnames = list(per_query_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_results)

    # Save readable text summary.
    with open(output_folder / "summary.txt", "w") as f:
        f.write("Adaptive Re-ranking results\n")
        f.write("---------------------------\n")
        f.write(recalls_str + "\n\n")

        f.write("Adaptive K statistics\n")
        f.write("---------------------\n")
        f.write(f"low threshold percentile {args.low_percentile:.1f}: {low_threshold:.6f}\n")
        f.write(f"high threshold percentile {args.high_percentile:.1f}: {high_threshold:.6f}\n")
        f.write(f"Average K: {avg_k:.2f}\n")

        for k in k_values:
            percentage = np.mean(used_ks == k) * 100
            f.write(f"Queries with K={k}: {percentage:.1f}%\n")

        f.write(
            f"\nAverage local matching reduction compared to fixed Top-{max_num_preds}: "
            f"{reduction:.1f}%\n"
        )

    print(f"\nResults saved in: {output_folder}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
