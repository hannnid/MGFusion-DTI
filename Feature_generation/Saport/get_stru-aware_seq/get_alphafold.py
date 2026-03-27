import requests
import os
import json
import sys

def download_alphafold_models(uniprot_ids, output_dir, failed_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to keep track of failed downloads
    failed_downloads = []

    for uniprot_id in uniprot_ids:
        api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}'
        response = requests.get(api_url)

        if response.status_code == 200:
            try:
                data = json.loads(response.content)
                if 'cifUrl' in data[0]:
                    cif_url = data[0]['cifUrl']
                    cif_response = requests.get(cif_url, stream=True)

                    if cif_response.status_code == 200:
                        cif_path = os.path.join(output_dir, f'{uniprot_id}.cif')
                        with open(cif_path, 'wb') as f:
                            f.write(cif_response.content)
                        print(f"Downloaded {uniprot_id}.cif")
                    else:
                        print(f"Failed to download CIF file from {cif_url}")
                        failed_downloads.append(uniprot_id)
                else:
                    print(f"No CIF URL found for {uniprot_id}")
                    failed_downloads.append(uniprot_id)

            except json.JSONDecodeError:
                print(f"Invalid JSON response for {uniprot_id}")
                failed_downloads.append(uniprot_id)
        else:
            print(f"Failed to get data for {uniprot_id}, HTTP Status Code: {response.status_code}")
            failed_downloads.append(uniprot_id)

    # Write failed downloads to a file
    with open(failed_path, 'w') as file:
        for id in failed_downloads:
            file.write(f"{id}\n")

    return failed_downloads


# Select dataset: BioSNAP, DrugBank, BindingDB, Human
dataset = "BioSNAP"
if len(sys.argv) > 1:
    dataset = sys.argv[1]

# Build dynamic paths
base_path = f"../../../Datasets/{dataset}/feature"
input_path = os.path.join(base_path, "protein_list.txt")
output_dir = os.path.join(base_path, "CIF_AF2")
failed_path = os.path.join(base_path, "failed_downloads.txt")

# Read UniProt IDs (first column of each line)
with open(input_path, 'r') as file:
    uniprot_ids = [line.strip().split()[0] for line in file if line.strip()]

# Run downloader
failed_ids = download_alphafold_models(uniprot_ids, output_dir, failed_path)

print(f"Failed downloads: {failed_ids}")