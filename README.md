# MGFusion-DTI
### MGFusion-DTI: Structure-Aware Multi-Granularity Fusion for Cold-Start DTI Prediction
MGFusion-DTI is a structure-aware multi-granularity fusion framework for drug-target interaction (DTI) prediction under cold-start scenarios. The framework incorporates protein structural information, binding pocket priors, pocket-level token interaction, and global-level representation fusion to improve generalization to unseen drugs and targets.

# 🧠 Overview
Drug-target interaction (DTI) prediction plays an important role in computational drug discovery and drug repurposing. However, existing methods often suffer from limited generalization ability under cold-start scenarios, where unseen drugs, unseen proteins, or both appear during testing. To address this challenge, MGFusion-DTI combines structure-aware protein representations with multi-granularity interaction modeling.

### ✨ Key Features
* Structure-aware protein representation using AlphaFold structures, Foldseek, and SaProt
* Binding pocket residue extraction from protein 3D structures
* Drug substructure representation using Mol2Vec
* Pocket-level token interaction via cross-attention
* Global-level drug-protein representation fusion via a Bi-Gated Fusion module
* Evaluation under warm-start, drug cold-start, target cold-start, and blind-start settings

### 🧩 MGFusion-DTI framework

<div align="center">
<p><img src="framework.jpg" width="700" /></p>
</div>


# ⚙️ Installation
```
# Clone the repository
git clone https://github.com/hannnid/MGFusion-DTI.git
cd MGFusion-DTI

# Create a conda environment
conda create -n MGFusion-DTI python=3.8.0
conda activate MGFusion-DTI

# Install bio_embeddings
pip install bio-embeddings==0.2.2
pip install bio-embeddings[all]

# Install PyTorch according to your CUDA version
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# or CUDA 10.2
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

# 📊 Resources
Repository Structure
```
MGFusion-DTI/
├── Datasets/
│   └── {dataset}/
│       ├── feature/
│       └── ...
├── Feature_generation/
│   ├── Mol2Vec/
│   ├── SaProt/
│   └── ProPocket/
├── Train/
│   └── model/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── Results/
├── README.md
└── requirements.txt
```

### Dataset 
Due to size and licensing restrictions, datasets are not included. We use: BioSNAP，BindingDB，DrugBank.  
Please download the original datasets from the following sources:
* [BioSNAP](https://snap.stanford.edu/biodata/](https://github.com/kexinhuang12345/MolTrans))
* [DrugBank](https://go.drugbank.com/](https://github.com/lifanchen-simm/transformerCPI))
* [Human](https://go.drugbank.com/](https://github.com/lifanchen-simm/transformerCPI))  
After downloading, please place the processed dataset files under: Datasets/{dataset}/  

  
# 🔬 Feature Generation
The feature generation pipeline consists of three parts:  
1. Drug representation generation
2. Structure-aware protein representation generation
3. Binding pocket extraction

### 1. 💊Drug Representation with Mol2Vec  
We adopt a customized version of [Mol2Vec](https://github.com/samoturk/mol2vec)⁠￼ to encode drug molecules into substructure-level representations.
```
python Feature_generation/Mol2Vec/Mol2Vec.py --dataset {dataset}
```
The generated feature files will be saved in: Datasets/{dataset}/feature/  
### 2. 🧬Protein Representation with SaProt  
MGFusion-DTI constructs structure-aware protein representations using AlphaFold structures, Foldseek, and SaProt.  
👉Step 1: Obtain Protein Structures  
Download protein structure files from the [AlphaFold Protein Structure Database⁠](https://alphafold.ebi.ac.uk/)￼ using [UniProt](https://www.uniprot.org/) IDs.
```
python Feature_generation/SaProt/get_alphafold.py --dataset {dataset}
```
👉Step 2: Generate Structure-Aware Sequences  
Convert protein structures into structure-aware sequences using Foldseek.
```
python Feature_generation/SaProt/generate_stru_seq.py --dataset {dataset}
```
👉Step 3: Encode Structure-Aware Sequences with SaProt  
Encode structure-aware protein sequences into residue-level embedding matrices.
```
python Feature_generation/SaProt/SaProt.py --dataset {dataset}
```
The generated feature file will be saved in: Datasets/{dataset}/feature/  
###### ⚠️Important Notes
The Foldseek binary is required but not included in this repository due to file size limitations. Please download Foldseek and place the executable file in the corresponding utility folder before generating structure-aware sequences.  
### 3. 🪢Binding Pocket Extraction
To capture biologically relevant interaction regions, MGFusion-DTI extracts binding pocket residues from protein 3D structures.
```
python Feature_generation/ProPocket/generate_proteinPocket.py --dataset {dataset}
```
The generated pocket-aware protein feature file will be saved in: Datasets/{dataset}/feature/  

  
# 🏋️ Reproducibility with Training
This section describes how to reproduce the results by training MGFusion-DTI from scratch.

👉Step 1: Generate Feature Files  
1.1 Drug Features
```
python Feature_generation/Mol2Vec/Mol2Vec.py --dataset {dataset}
```
1.2 Protein Features
```
python Feature_generation/SaProt/SaProt.py --dataset {dataset}
```
1.3 Binding Pocket Features
```
python Feature_generation/ProPocket/generate_proteinPocket.py --dataset {dataset}
```
👉Step 2: Train and Evaluate  
The training and evaluation scripts are located in: Train/model/  
Run the following command:
```
cd Train/model
python train.py --datasets {dataset} --scenarios {scenario}
```
The results will be saved in: Train/model/Results/{dataset}/ 
###### 📊Supported datasets: BioSNAP, DrugBank, Human  
###### 🧪Supported scenarios: warm_start, drug_cold_start, target_cold_start, blind_start  
  
 
# 🚀 Reproducibility without Training
We also provide pretrained models for direct evaluation without retraining.

👉Step 1: Download Pretrained Models
Please download the pretrained models from: [Google Drive](https://drive.google.com/drive/folders/1yJ-bZHnjy45VyHmet1BlD9ah8szJiTLF?usp=sharing)  
After downloading, place the pretrained model folder under: Train/model/Results/{dataset}/  

👉Step 2: Prepare Feature Files
Please make sure that the following feature files are available in: Datasets/{dataset}/feature/  
Required feature files:
```
smiles_Atom2Vec300.pkl
smiles_Mol2Vec300.pkl
sa_SaProt1280.pkl
sa_SaProt1280_pockets.pkl
```

👉Step 3: Evaluate the Pretrained Model  
```
cd Train/model
python train.py --datasets {dataset} --scenarios {scenario}
```
The results will be saved in: Train/model/Results/{dataset}/  
###### 📊Supported datasets: BioSNAP, DrugBank, Human  
###### 🧪Supported scenarios: warm_start, drug_cold_start, target_cold_start, blind_start  


# 📬 Contact
If you have any questions, please feel free to contact us:  
👧Hui Han  
Department of Computer Science, University of Tsukuba  
Email: s2530161@u.tsukuba.ac.jp  


	
