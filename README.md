# MGFusion-DTI
MGFusion-DTI: Structure-Aware Multi-Granularity Fusion for Cold-Start DTI Prediction

Drug-target interaction (DTI) prediction plays a crucial role in computational drug discovery, facilitating drug repurposing and accelerating new drug development. Despite recent advances in deep learning-based methods, model performance remains limited under cold-start scenarios, where unseen drugs or targets are encountered. Moreover, existing approaches often rely on global-level feature fusion, which may overlook fine-grained local interactions driven by key residues and neglect the structural context of proteins. To address these challenges, we propose a structure-aware multi-granularity fusion framework, termed MGFusion-DTI, for DTI prediction under cold-start scenarios. The proposed model integrates structure-aware protein representations with sequence-based drug features and adopts a multi-level interaction modeling strategy. Specifically, a cross-attention mechanism is employed to capture fine-grained interactions between drug substructures and protein residues, with emphasis on residues located in predicted binding pockets. In parallel, global representations are jointly modeled to account for potential long-range regulatory effects, enabling the model to capture both local binding patterns and global structural dependencies. Extensive experiments on three benchmark datasets demonstrate that MGFusion-DTI consistently outperforms state-of-the-art methods across both warm-start and multiple cold-start scenarios, with particularly strong performance in blind-start settings.

## MGFusion-DTI framwork

<div align="center">
<p><img src="framwork.jpg" width="600" /></p>
</div>

## Contents
- [Installation](#Installation)
- [Demo data](#Demo-data)
- [Resources](#Resources)
- [Reproducibility](#Reproducibility)
- [Prediction](#Prediction)
- [Contact](#Contact)


## Installation

MGFusion-DTI is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).
   - Prerequisites: \
       [Python3.*](https://www.python.org/) (version>=3.8)\
	   [gensim](https://github.com/piskvorky/gensim.git) (version=3.8.3)\
       [Mol2Vec](https://github.com/samoturk/mol2vec) \
       [bio_embeddings](https://github.com/sacdallago/bio_embeddings) \
       [CUDA Toolkit](https://anaconda.org/anaconda/cudatoolkit) (version>=10.2, for GPU only)
   - Dependencies: \
       [PyTorch](https://pytorch.org/) (version >=1.9.0, <=1.16.0) \
	   [numpy](http://www.numpy.org/) (version = 1.22.0)\
	   [scikit-learn](https://scikit-learn.org/stable/) (version = 1.0.2)\
	   [pandas](https://github.com/pandas-dev/pandas) (version = 1.0.1)\
	   [rdkit](https://github.com/rdkit/rdkit) (version = 2022.9.4)\
	   [tqdm](https://github.com/tqdm/tqdm) \
	   [prefetch_generator](https://github.com/justheuristic/prefetch_generator) \

   - Installation typically requires around 1 to 2 hours, depending on network conditions.

#### System Requirements
`MGFusion-DTI` requires only a standard computer with enough RAM to support the in-memory operations. Using GPU could acceralate the training and inference of models.

Recommended Hardware: 128 GB RAM, 40 CPU processors, 4 TB disk storage, >=30 GB GPU 


#### Installation

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
  
## Resources
+ README.md: this file.
+ Datasets: The dataset used by MGFusion-DTI
	+ 「Datsetsname」: 
		+ warm_start: The datasets for warm start.
		+ compound_cold_start: The datasets for compound cold start.
		+ protein_cold_start: The datasets for protein cold start.
		+ blind_start: The datasets for blind start.
		+ feature: Contain the SMILES strings of drugs and structure-aware sequences of proteins. 
			+ drug_list.txt: The SMILES strings of drugs
			+ protein_list.txt: SA sequences of proteins
		+ drug_without_feature.txt: Contain the drugs of which the SMILES cannot be recongnized by Mol2Vec.
		+ full_pair.txt: The full dataset with positives and negatives for performance evaluation.
		+ protein_without_feature.txt: Contain the proteins without 3D files.

+ Feature_generation

	+ Mol2Vec
	
	Mol2Vec is customised version of Mol2Vec(https://github.com/samoturk/mol2vec). We recode the mol2vec/feature.py to generate feature matrices of compounds.
	
	You will obtain the feature vectors and matrices of the compounds by following command. **dataname** should be BindingDB_AIBind, BioSNAP, or BindingDB.

		python Mol2Vec.py --dataset dataname
	
	+ Saport
	
	You will obtain the feature vectors and matrices of the proteins by following command. **dataname** should be BindingDB_AIBind, BioSNAP, or BindingDB.
	
		python generator.py --dataset dataname
		
+ Pretrian_models
		
	+ BindingDB_AIBind: The trained models on BindingDB_AIbind dataset under warm start, compound cold start, protein cold start, and blind start.
	
	+ BioSNAP: The trained models on BioSNAP dataset under warm start, compound cold start, protein cold start, and blind start.
	
	+ BindingDB: The trained models on BindingDB dataset under warm start, compound cold start, protein cold start, and blind start.
	
	
+ Train
	+ MGFusion-DTI: The codes of training, testing, and model.
		+ ablation
			+ model.py: The codes of WOPretrain, WODecouple, WOTransformer, MolTrans_pretrain, and DrugBAN_pretrain.
			+ dataset.py
			+ train_decouple.py: The code of evaluation of WODecouple .
			+ train_transformer.py: The code of evaluation of WOTransformer.
			+ train_wopretrain.py: The code of evaluation of WOPretrain.
			+ train_DrugBAN_pretrain.py: The code of evaluation of DrugBAN_pretrain.
			+ train_MolTrans_pretrain.py: The code of evaluation of MolTrans_pretrain.
		+ dataset.py
		+ model.py: The code of MGFusion-DTI.
		+ train_BindingDB_AIBind.py: The code of evaluation in BindingDB_AIBind under warm start, compound cold start, protein cold start, and blind start.
		+ train_BindingDB_AIBind_missing.py: The code of evaluation in BindingDB_AIBind with scarce data.
		+ train_BindingDB.py: The code of evaluation in BindingDB under warm start, compound cold start, protein cold start, and blind start.
		+ train_BindingDB_missing.py: The code of evaluation in BindingDB with scarce data.
		+ train_BioSNAP.py: The code of evaluation in BioSNAP under warm start, compound cold start, protein cold start, and blind start.
		+ train_BioSNAP_missing.py: The code of evaluation in BioSNAP with scarce data.

+ Case study: The raw files(PDB and pdbqt), settings and results of Docking.

+ Demo: The code and data for demo.

+ Predictions: 
Provides trained models and scripts to predict CPIs between user-submitted compound libraries and protein libraries.
	+ Custom_Data: Reference (default) data
		+ default
			+ drug_list.txt: Standard format for compound libraries.
			+ protein_list.txt: Standard format for protein libraries.
	+ checkpoint.pth: Trained model.
	+ Mol2Vec
	+ predictor.py: Prediction Script.
	+ model.py
	+ dataset.py

+ Source_Data: Source data and code used in the manuscript to plot individual figures and tables.


## Reproducibility

### Reproducibility with training

For the warm start experiment on the BindingDB_AIBind dataset, you can directly run the following setps.

+ step 1: Generate the feature matrices of compounds and proteins
	+ 1.1 For compounds:
	
		+ python Mol2Vec.py --dataset BindingDB_AIBind
		
		The compound_Mol2Vec300.pkl and compound_Atom2Vec300.pkl will generated in [_feature_](/Datasets/BindingDB_AIBind/feature).
		
	+ 1.2 For proteins:
		+ python generator.py --dataset BindingDB_AIBind
		
		The aas_ProtTransBertBFD1024.pkl will generated in [_feature_](/Datasets/BindingDB_AIBind/feature).
		
+ setp 2: Training and testing. The codes are in the [_Train/MGFusion-DTI_](/Train/MGFusion-DTI) folder.

	+ python train_BindingDB_AIBind.py --scenarios warm_start
	
	The results are saved in the [_Results_](/Train/MGFusion-DTI/Results) folder.
	
### Reproducibility without training

We also provide models that have been trained for direct testing. For the warm start experiment on the BindingDB_AIBind dataset, you can directly run the following setps.

+ step 1: Make sure that [_feature_](/Datasets/BindingDB_AIBind/feature) already holds the pre-training feature files (i.e., compound_Atom2Vec300.pkl, compound_Mol2Vec300.pkl, and aas_ProtTransBertBFD1024.pkl) for compounds and proteins;

+ setp 2: Move the fold [_BindingDB_AIBind_](/Pretrian_models/BindingDB_AIBind) to the [_Results_](/Train/MGFusion-DTI/Results) folder; 

+ setp 3: Loading trained model and testing

	+ python train_BindingDB_AIBind.py --scenarios warm_start
	
The results are saved in the [_Results_](/Train/MGFusion-DTI/Results) folder.


## Contact

If any questions, please do not hesitate to contact us at:

Hui Han, s2530161@u.tsukuba.ac.jp


		

	
