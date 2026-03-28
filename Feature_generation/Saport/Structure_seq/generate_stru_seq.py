import os
import pandas as pd
from utils.util_foldseek import get_struc_seq

dataset = "Human"  # Human BioSNAP, DrugBank, BindingDB
base_dir = f"../../../Datasets/{dataset}/feature"

def process_cif_files(directory, batch_size=100):
    global base_dir
    seq_data = []
    combined_seq_data = []
    failed_ids = []  # Store IDs for which data could not be retrieved
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            path = os.path.join(directory, filename)
            uniprot_id = filename.split('.')[0]
            chain_found = False  # Flag to indicate if a valid chain is found

            for chain in [chr(i) for i in range(ord('A'), ord('Z')+1)]:
                try:
                    parsed_seqs = get_struc_seq("./utils/foldseek", path, [chain])

                    if chain in parsed_seqs:
                        seq, _, combined_seq = parsed_seqs[chain]
                        seq_data.append({'uniprot_id': uniprot_id, 'seq': seq})
                        combined_seq_data.append({'uniprot_id': uniprot_id, 'combined_seq': combined_seq})

                        count += 1
                        if count % batch_size == 0:
                            save_to_csv(seq_data, combined_seq_data, append=True)
                            seq_data = []
                            combined_seq_data = []

                        chain_found = True
                        break
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

            if not chain_found:
                failed_ids.append(uniprot_id)

    # Save remaining data
    if seq_data or combined_seq_data:
        save_to_csv(seq_data, combined_seq_data, append=True)

    # Write failed UniProt IDs to a file
    if failed_ids:
        with open(os.path.join(base_dir, "failed_UniProt_ids.txt"), 'w') as f:
            for id in failed_ids:
                f.write(f"{id}\n")

def save_to_csv(seq_data, combined_seq_data, append=False):
    sa_path = os.path.join(base_dir, "protein_list_sa.txt")
    seq_df = pd.DataFrame(seq_data)
    combined_seq_df = pd.DataFrame(combined_seq_data)

    # append SA sequences (id combined_seq per line)
    with open(sa_path, "a" if append else "w") as f:
        for _, row in combined_seq_df.iterrows():
            f.write(f"{row['uniprot_id']} {row['combined_seq']}\n")

    # save merged sequence CSV
    combined_path = os.path.join(base_dir, "Alpha_stru_seq.csv")
    merged_df = pd.merge(seq_df, combined_seq_df, on="uniprot_id")
    merged_df.to_csv(combined_path, mode="a" if append else "w",
                     index=False, header=not append)

directory = os.path.join(base_dir, "CIF_AF2")
process_cif_files(directory, batch_size=100)
