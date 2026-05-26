import os
import sys
import argparse
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="directory with predictions of a VPR model",
    )

    parser.add_argument(
        "--z-data",
        type=str,
        required=True,
        help="path to z_data.torch saved by main.py with --save_for_uncertainty",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory of adaptive image matching results",
    )

    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--im-size",
        type=int,
        default=512,
        help="resize img to im_size x im_size",
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="adaptive K values: small, medium, large",
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=35.0,
        help="lower percentile for FAISS gap",
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=70.0,
        help="higher percentile for FAISS gap",
    )

    parser.add_argument(
        "--start-query",
        type=int,
        default=-1,
        help="query to start from",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=-1,
        help="number of queries",
    )

    return parser.parse_args()


def choose_adaptive_k(gap, low_threshold, high_threshold, k_values):
    """
    Choose the number of candidates to match according to the uncertainty
    of the global retrieval stage.

    FAISS uses L2 distance in the current pipeline:
    - lower distance means better candidate;
    - gap = distance_2 - distance_1.

    A large gap means that the first candidate is clearly better than the
    second one, so the query is considered reliable and a smaller K is used.

    A small gap means that the first candidates are close to each other,
    so the query is uncertain and a larger K is used.
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
    z_data_path = Path(args.z_data)

    matcher_name = args.matcher
    device = args.device
    img_size = args.im_size
    k_values = args.k_values

    if len(k_values) != 3:
        raise ValueError("--k-values must contain exactly three values, for example: 5 10 20")

    matcher = get_matcher(matcher_name, device=device)

    output_folder = (
        Path(preds_folder + f"_{matcher_name}_adaptive")
        if args.out_dir is None
        else Path(args.out_dir)
    )
    output_folder.mkdir(exist_ok=True)

    z_data = torch.load(z_data_path, weights_only=False)
    faiss_distances = z_data["distances"]

    gaps = faiss_distances[:, 1] - faiss_distances[:, 0]

    low_threshold = np.percentile(gaps, args.low_percentile)
    high_threshold = np.percentile(gaps, args.high_percentile)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    if len(txt_files) != len(gaps):
        raise ValueError(
            f"Number of prediction txt files ({len(txt_files)}) is different from "
            f"number of queries in z_data ({len(gaps)}). Check that preds-dir and z-data belong to the same run."
        )

    start_query = args.start_query if args.start_query >= 0 else 0
    num_queries = args.num_queries if args.num_queries >= 0 else len(txt_files)

    selected_txt_files = txt_files[start_query:start_query + num_queries]

    used_ks = []

    for local_idx, txt_file in enumerate(tqdm(selected_txt_files, desc="Adaptive matching")):
        real_query_idx = start_query + local_idx

        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")

        gap = gaps[real_query_idx]
        adaptive_k = choose_adaptive_k(gap, low_threshold, high_threshold, k_values)
        used_ks.append(adaptive_k)

        if out_file.exists():
            continue

        q_path, pred_paths = read_file_preds(txt_file)

        if len(pred_paths) < adaptive_k:
            raise ValueError(
                f"{txt_file} contains only {len(pred_paths)} predictions, "
                f"but adaptive K is {adaptive_k}."
            )

        results = []

        img0 = matcher.load_image(q_path, resize=img_size)

        for pred_path in pred_paths[:adaptive_k]:
            img1 = matcher.load_image(pred_path, resize=img_size)

            result = matcher(deepcopy(img0), img1)

            # Avoid saving heavy descriptors.
            result["all_desc0"] = None
            result["all_desc1"] = None

            results.append(result)

        torch.save(results, out_file)

    used_ks = np.array(used_ks)

    print("\nAdaptive matching completed")
    print("---------------------------")
    print(f"Output folder: {output_folder}")
    print(f"low threshold  percentile {args.low_percentile:.1f}:  {low_threshold:.6f}")
    print(f"high threshold percentile {args.high_percentile:.1f}: {high_threshold:.6f}")
    print(f"Average K: {used_ks.mean():.2f}")

    for k in k_values:
        percentage = np.mean(used_ks == k) * 100
        print(f"Queries with K={k}: {percentage:.1f}%")

    print(
        f"Average matching reduction compared to fixed Top-{max(k_values)}: "
        f"{(1 - used_ks.mean() / max(k_values)) * 100:.1f}%"
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
