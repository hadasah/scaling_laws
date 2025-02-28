""" 
Quick wikipedia download script from huggingface for quickstart purposes.
Just downloads the 20220301 english wikipedia from huggingface and 
does no extra preprocessing.

"""

import argparse
from datasets import load_dataset  # huggingface
import os


def main(output_dir, dataset, dataset_version, train_data_files, valid_data_files, version, overwrite=False):
    dataset_str = f"{dataset}_{dataset_version}" if dataset_version else dataset
    output_dir = os.path.join(output_dir, dataset_str, version)
    
    if not overwrite and (
        os.path.isdir(output_dir) and
        len(os.listdir(output_dir)) != 0
    ):
        return
    os.makedirs(output_dir, exist_ok=True)
    data = load_dataset(
        dataset, 
        dataset_version, 
        data_files={
            "train": train_data_files.split(','),
            "valid": valid_data_files.split(','),
            } if train_data_files else None,
    )

    for split, dataset in data.items():
        print("Processing split: %s" % data)
        output_file = os.path.join(output_dir, f"{str(split)}.jsonl")
        dataset.to_json(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store the .jsonl file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="hf dataset name",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        help="hf dataset version",
    )
    parser.add_argument(
        "--train-data-files",
        type=str,
        default=None,
        help="hf dataset files",
    )
    parser.add_argument(
        "--valid-data-files",
        type=str,
        default=None,
        help="hf dataset files",
    )
    parser.add_argument(
        "--version-name",
        type=str,
        help="internal processed version name",
    )

    args = parser.parse_args()

    main(args.output_dir, args.dataset, args.dataset_version, args.train_data_files, args.valid_data_files, args.version_name)
