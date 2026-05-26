import numpy as np
from tqdm import tqdm
import os
import argparse
from glob import glob
from pathlib import Path
import torch

from util import get_list_distances_from_preds


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="directory with predictions of a VPR model",
    )

    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="directory with image matching results",
    )

    parser.add_argument(
        "--z-data",
        type=str,
        required=True,
        help="path to z_data.torch saved by main.py with --save_for_uncertainty",
    )

    parser.add_argument(
        "--max-num-preds",
        type=int,
        default=20,
        help="maximum number of predictions available for adaptive re-ranking",
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="adaptive K values: small, medium, large",
    )

    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance in meters for a prediction to be considered positive",
    )

    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="recall values to compute",
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=35.0,
        help="lower percentile for the uncertainty gap",
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=70.0,
        help="higher percentile for the uncertainty gap",
    )

    return parser.parse_args()


def choose_adaptive_k(gap, low_threshold, high_threshold, k_values):
    """
    Choose the number of candidates to re-rank according to retrieval uncertainty.

    FAISS uses L2 distance:
    - lower distance means better candidate;
    - gap = distance_2 - distance_1.

    Large gap  -> confident query   -> smaller K.
    Medium gap -> medium uncertainty -> intermediate K.
    Small gap  -> uncertain query   -> larger K.
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

    max_num_preds = args.max_num_preds
    k_values = args.k_values

    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    if len(k_values) != 3:
        raise ValueError("--k-values must contain exactly three values, for example: 5 10 20")

    if max(recall_values) > max_num_preds:
        raise ValueError("max recall value cannot be larger than --max-num-preds")

    z_data = torch.load(z_data_path, weights_only=False)
    faiss_distances = z_data["distances"]

    if faiss_distances.shape[1] < max_num_preds:
        raise ValueError(
            f"z_data contains only {faiss_distances.shape[1]} predictions, "
            f"but --max-num-preds is {max_num_preds}."
        )

    gaps = faiss_distances[:, 1] - faiss_distances[:, 0]

    low_threshold = np.percentile(gaps, args.low_percentile)
    high_threshold = np.percentile(gaps, args.high_percentile)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)

    if total_queries != len(gaps):
        raise ValueError(
            f"Number of txt prediction files ({total_queries}) is different from "
            f"number of queries in z_data ({len(gaps)}). Check that preds-dir and z-data belong to the same run."
        )

    recalls = np.zeros(len(recall_values))
    used_ks = []

    for query_idx, txt_file_query in enumerate(tqdm(txt_files, desc="Adaptive re-ranking")):
        original_geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:max_num_preds]

        gap = gaps[query_idx]
        adaptive_k = choose_adaptive_k(gap, low_threshold, high_threshold, k_values)
        used_ks.append(adaptive_k)

        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace(".txt", ".torch"))

        if not torch_file_query.exists():
            raise FileNotFoundError(
                f"Missing matching file: {torch_file_query}\n"
                f"Check if the inliers directory is correct."
            )

        query_results = torch.load(torch_file_query, weights_only=False)

        if len(query_results) < adaptive_k:
            raise ValueError(
                f"{torch_file_query} contains only {len(query_results)} matching results, "
                f"but adaptive K is {adaptive_k}."
            )

        query_db_inliers = torch.zeros(adaptive_k, dtype=torch.float32)

        for i in range(adaptive_k):
            query_db_inliers[i] = query_results[i]["num_inliers"]

        _, reranked_indices = torch.sort(query_db_inliers, descending=True)

        # The first adaptive_k candidates are re-ranked.
        # The remaining candidates are appended in the original global retrieval order.
        # This keeps Recall@10 and Recall@20 meaningful even when adaptive_k is 5 or 10.
        remaining_indices = torch.arange(adaptive_k, max_num_preds)

        final_indices = torch.cat([reranked_indices, remaining_indices])

        final_geo_dists = original_geo_dists[final_indices]

        for i, n in enumerate(recall_values):
            if torch.any(final_geo_dists[:n] <= threshold):
                recalls[i:] += 1
                break

    recalls = recalls / total_queries * 100

    used_ks = np.array(used_ks)
    avg_k = used_ks.mean()

    print("\nAdaptive Re-ranking results")
    print("---------------------------")

    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)]
    )
    print(recalls_str)

    print("\nAdaptive K statistics")
    print("---------------------")
    print(f"low threshold  percentile {args.low_percentile:.1f}:  {low_threshold:.6f}")
    print(f"high threshold percentile {args.high_percentile:.1f}: {high_threshold:.6f}")
    print(f"Average K: {avg_k:.2f}")

    for k in k_values:
        percentage = np.mean(used_ks == k) * 100
        print(f"Queries with K={k}: {percentage:.1f}%")

    print(
        f"\nAverage local matching reduction compared to fixed Top-{max_num_preds}: "
        f"{(1 - avg_k / max_num_preds) * 100:.1f}%"
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
