
  + datsetsname: 
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

