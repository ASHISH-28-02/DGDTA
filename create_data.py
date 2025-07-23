import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from utils import TestbedDataset # Assuming this is in utils.py

# --- Helper Functions (No changes here) ---

def get_atom_features(atom):
    return np.array(
        encode_one_hot_unknown(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                                 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                                                 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                                                 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        encode_one_hot(atom.GetDegree(), list(range(11))) +
        encode_one_hot_unknown(atom.GetTotalNumHs(), list(range(11))) +
        encode_one_hot_unknown(atom.GetImplicitValence(), list(range(11))) +
        [atom.GetIsAromatic()]
    )

def encode_one_hot(value, allowable_set):
    if value not in allowable_set:
        raise ValueError(f"Input {value} not in allowable set {allowable_set}")
    return [value == s for s in allowable_set]

def encode_one_hot_unknown(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == s for s in allowable_set]

def convert_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    atom_count = mol.GetNumAtoms()
    features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    graph = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in graph.edges]
    return atom_count, features, edge_index

def encode_sequence(protein, seq_dict, max_seq_len):
    encoded_seq = np.zeros(max_seq_len)
    for i, char in enumerate(protein[:max_seq_len]):
        if char in seq_dict: encoded_seq[i] = seq_dict[char]
    return encoded_seq


# --- Main Processing Logic (Corrected) ---

def process_dataset(dataset_name):
    print(f'Converting raw data for {dataset_name} into intermediate CSV files...')
    data_path = f'data/{dataset_name}/'

    # Load raw data
    ligands = json.load(open(data_path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(data_path + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinities = pickle.load(open(data_path + "Y", "rb"), encoding='latin1')

    if dataset_name == 'davis':
        affinities = [-np.log10(y / 1e9) for y in affinities]
    affinities = np.asarray(affinities)

    # =======================================================
    # THE CORE FIX: Correctly interpret the indices
    # =======================================================
    # 1. Find the coordinates of all valid (not NaN) drug-target pairs in the affinity matrix
    valid_drug_indices, valid_protein_indices = np.where(~np.isnan(affinities))
    
    # 2. Load the indices for the train/test splits. These are indices INTO the lists of valid pairs.
    train_fold_indices = [idx for sublist in json.load(open(data_path + "folds/train_fold_setting1.txt")) for idx in sublist]
    test_fold_indices = json.load(open(data_path + "folds/test_fold_setting1.txt"))
    
    drug_keys = list(ligands.keys())
    protein_keys = list(proteins.keys())
    
    for phase, fold_indices in [('train', train_fold_indices), ('test', test_fold_indices)]:
        df_data = []
        # 3. Iterate through the indices for the current fold (train or test)
        for i in fold_indices:
            # 4. Use the index 'i' to get the correct drug and protein index from the valid pair lists
            drug_idx = valid_drug_indices[i]
            protein_idx = valid_protein_indices[i]
            
            # 5. Look up the data using these correct indices
            drug_smiles = ligands[drug_keys[drug_idx]]
            protein_sequence = proteins[protein_keys[protein_idx]]
            affinity_value = affinities[drug_idx, protein_idx]
            
            # Canonicalize SMILES to ensure consistency
            canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(drug_smiles), isomericSmiles=True)
            df_data.append([canon_smiles, protein_sequence, affinity_value])

        df = pd.DataFrame(df_data, columns=['compound_iso_smiles', 'target_sequence', 'affinity'])
        df.to_csv(f'data/{dataset_name}_{phase}.csv', index=False)
        print(f'Created data/{dataset_name}_{phase}.csv with {len(df)} entries.')

# --- Script Execution ---

if __name__ == "__main__":
    sequence_vocabulary = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    sequence_dict = {char: (i + 1) for i, char in enumerate(sequence_vocabulary)}
    max_sequence_length = 1000
    
    datasets_to_process = ['davis']
    
    # Step 1: Create the intermediate CSV files
    for dataset in datasets_to_process:
        process_dataset(dataset)
    
    # Step 2: Collect all unique drug SMILES
    print("\nCollecting all unique SMILES strings...")
    compound_smiles = set()
    for dataset in datasets_to_process:
        for phase in ['train', 'test']:
            df = pd.read_csv(f'data/{dataset}_{phase}.csv')
            compound_smiles.update(df['compound_iso_smiles'])
    print(f"Found {len(compound_smiles)} unique SMILES strings.")

    # Step 3: Pre-compute graphs for all unique SMILES
    print("Pre-computing graphs for all SMILES strings...")
    smile_graphs = {smile: convert_smile_to_graph(smile) for smile in compound_smiles if convert_smile_to_graph(smile)[0] is not None}
    print("Graph pre-computation complete.")

    # Step 4: Create the final .pt files for PyTorch
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)

    for dataset in datasets_to_process:
        print(f'\nCreating PyTorch Geometric files for {dataset}...')
        train_df = pd.read_csv(f'data/{dataset}_train.csv')
        train_drugs, train_proteins, train_affinities = list(train_df['compound_iso_smiles']), list(train_df['target_sequence']), list(train_df['affinity'])
        train_proteins_encoded = [encode_sequence(p, sequence_dict, max_sequence_length) for p in train_proteins]

        test_df = pd.read_csv(f'data/{dataset}_test.csv')
        test_drugs, test_proteins, test_affinities = list(test_df['compound_iso_smiles']), list(test_df['target_sequence']), list(test_df['affinity'])
        test_proteins_encoded = [encode_sequence(p, sequence_dict, max_sequence_length) for p in test_proteins]

        print(f'Preparing {dataset}_train.pt...')
        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train', xd=train_drugs, xt=train_proteins_encoded, y=train_affinities, smile_graph=smile_graphs)
        
        print(f'Preparing {dataset}_test.pt...')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test', xd=test_drugs, xt=test_proteins_encoded, y=test_affinities, smile_graph=smile_graphs)