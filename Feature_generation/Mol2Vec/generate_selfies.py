# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import selfies as sf
from pandarallel import pandarallel
from tqdm import tqdm
import os


def to_selfies(smiles):
    try:
        if smiles is None:
            return None
        return sf.encoder(smiles)
    except sf.EncoderError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="BioSNAP",
        choices=["BioSNAP", "BindingDB_AIBind", "BindingDB"],
        help="Name of the dataset "
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/work/data1/hanhui/ColdstartCPI_1/Datasets",
        help="Root directory where dataset folders are stored"
    )
    args = parser.parse_args()

    dataset = args.dataset
    feature_dir = args.feature_dir

    input_path = os.path.join(feature_dir, dataset, "feature", "drug_list.txt")
    output_path = os.path.join(feature_dir, dataset, "feature", "drug_list_selfies.txt")

    assert os.path.exists(input_path), f"Input file not found: {input_path}"
    print(f"Reading from: {input_path}")

    # 读取 drug_list.txt
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                drug_id, smiles = parts[0], parts[1]
                data.append((drug_id, smiles))

    df = pd.DataFrame(data, columns=["drug_id", "smiles"])

    # 初始化并并行转换为 SELFIES
    pandarallel.initialize()
    tqdm.pandas(desc="Converting SMILES to SELFIES")
    df["selfies"] = df["smiles"].parallel_apply(to_selfies)

    failed_df = df[df["selfies"].isna()]
    if not failed_df.empty:
        print("❌ Failed to convert the following drug IDs to SELFIES:")
        for drug_id in failed_df["drug_id"]:
            print(f"{drug_id}")

    # 删除无法转换的行
    df.dropna(subset=["selfies"], inplace=True)

    # 保存为 drug_list_selfies.txt，空格分隔
    with open(output_path, 'w') as f:
        for drug_id, selfies in zip(df["drug_id"], df["selfies"]):
            f.write(f"{drug_id} {selfies}\n")

    print(f"Successfully converted {len(df)} entries.")
    print(f"Saved SELFIES to: {output_path}")


if __name__ == "__main__":
    main()