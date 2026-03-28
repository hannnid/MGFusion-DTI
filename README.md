# MGFusion-DTI
MGFusion-DTI: Structure-Aware Multi-Granularity Fusion for Cold-Start DTI Prediction

## 🧠 Introduction
Drug-Target Interaction (DTI) prediction plays a crucial role in drug discovery and repositioning. However, existing methods often suffer from limited generalization ability, especially in cold-start scenarios, where unseen drugs or proteins appear during testing.  

To address this challenge, we propose MGFusion-DTI, a novel framework that:  
	•	Incorporates structure-aware sequence representations  
	•	Extracts binding pocket residues from protein structures  
	•	Performs multi-granularity fusion at both:  
	•	Token-level (matrix-level) via cross-attention  
	•	Vector-level (global-level) via gated fusion  
	•	Enhances generalization performance in cold-start settings  

## 🧩 MGFusion-DTI framwork

<div align="center">
<p><img src="framwork.jpg" width="600" /></p>
</div>


## ⚙️ Installation

```shell
# download MGFusion-DTI
git clone https://github.com/hannnid/MGFusion-DTI
cd MGFusion-DTI

# create environment named MGFusion-DTI
conda create -n MGFusion-DTI python=3.8.0

# then the environment can be activated to use
conda activate MGFusion-DTI

# install bio_embeddings
pip install bio-embeddings==0.2.2
pip install bio-embeddings[all]

# Install pytorch according to hardware
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# or
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch

# install other tools in requirements.txt
pip install -r requirements.txt

```
	
## 📊 Resources
+ 🔹README.md: this file.
+ 🔹Datasets: The dataset used by MGFusion-DTI
	Due to size and licensing restrictions, datasets are not included. We use: BioSNAP，BindingDB，DrugBank.
	Please download from: [BioSNAP](https://snap.stanford.edu/biodata/), [BindingDB](https://www.bindingdb.org/), [Human](https://go.drugbank.com/)
	
+ 🔹Feature Generation  
  The feature generation pipeline consists of two branches: drug representation and protein representation, followed by binding pocket extraction.  
	+ Mol2Vec(💊Drug Representation)  
   We adopt a customized version of [Mol2Vec](https://github.com/samoturk/mol2vec) to encode drug molecules.  
    (dataname ∈ {BioSNAP, BindingDB, Human}), The generated features are stored in the corresponding dataset folder.
   
			python Mol2Vec.py --dataset dataname  
	+ Saprot(🧬Protein Representation)  
   	We construct structure-aware protein representations based on 3D structure using AlphaFold + Foldseek + SaProt.  
		📌 Step 1: Obtain Protein Structures
		Download protein structure files (.cif) from: [AlphafoldDB](https://alphafold.ebi.ac.uk/)  
    	Using UniProt IDs from: [UniProt](https://www.uniprot.org/)

			python get_alphafold.py
  		📌 Step 2: Generate Structure-Aware Sequences  
		Convert protein structures into structure-aware sequences using Foldseek:

			python generate_stru_seq.py
 		📌 Step 3: Encode with SaProt  
 		Encode the structure-aware sequences into embedding matrices:  
	
 			python Saprot.py  
		⚠️ Important Notes
		•	The Foldseek binary is required but not included due to size limitations.You can download the binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view) and place it in the utils folder.  

  + ProPocket  
 	To capture biologically meaningful interaction regions, we extract binding pocket residues from protein 3D structures.

			python generate_proteinPocket.py --dataset dataname

+ 🔹Train
	+ model: The codes of training, testing, and model.
		+ dataset.py
		+ model.py: The code of MGFusion-DTI.
		+ train.py: The code of evaluation in Datasets under warm start, compound cold start, protein cold start, and blind start.

### 🏋️ Reproducibility with training

For the warm start experiment on the BioSNAP dataset, you can directly run the following setps.

+ step 1: Generate the feature matrices of drugs and proteins
	+ 1.1 For drugs:
	
		+ python Mol2Vec.py --dataset BioSNAP
		
		The smiles_Mol2Vec300.pkl and smiles_Atom2Vec300.pkl will generated in [_feature_](/Datasets/BioSNAP/feature).
		
	+ 1.2 For proteins:
		+ python generator.py --dataset BioSNAP
		
		The sa_SaProt1280.pkl will generated in [_feature_](/Datasets/BioSNAP/feature).

		+ python generate_proteinPocket.py --dataset BioSNAP
    
		The sa_SaProt1280_pockets.pkl will generated in [_feature_](/Datasets/BioSNAP/feature).  
+ setp 2: Training and testing. The codes are in the [_Train/model_](/Train/model) folder.

	+ python train.py --datasets BioSNAP --scenarios warm_start
	
	The results are saved in the [_Results_](/Train/model/Results) folder.
	
### Reproducibility without training

We also provide models that have been trained for direct testing. For the warm start experiment on the BioSNAP dataset, you can directly run the following setps.

+ step 1: Make sure that [_feature_](/Datasets/BioSNAP/feature) already holds the pre-training feature files (i.e., compound_Atom2Vec300.pkl, compound_Mol2Vec300.pkl, sa_SaProt1280.pkl and sa_SaProt1280_pockets.pkl) for drugs and proteins;

+ setp 2: Move the fold [_BindingDB_AIBind_](/Pretrian_models/BindingDB_AIBind) to the [_Results_](/Train/MGFusion-DTI/Results) folder; 

+ setp 3: Loading trained model and testing

	+ python train_BindingDB_AIBind.py --scenarios warm_start
	
The results are saved in the [_Results_](/Train/model/Results) folder.


## Contact

If any questions, please do not hesitate to contact us at:

Hui Han, s2530161@u.tsukuba.ac.jp


		

	
