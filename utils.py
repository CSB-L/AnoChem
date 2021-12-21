from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import argparse
import scaffoldgraph as sg

def argument_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--smiles_input', required=True, help="Input Entrez gene ID")
    parser.add_argument('-s', '--scaffold_flag', required=False, help="Domain change flag", action='store_true')
    return parser

def calculate_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        bits = fp.ToBitString()
        feat = []
        for f in bits:
            feat.append(int(f))
        feat_list.append(feat)
    return np.asarray(feat_list)

def get_scaffolds(smi):
    mol = Chem.MolFromSmiles(smi)
    frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False)
    frags_smiles = []
    for mol in frags:
        smiles = Chem.MolToSmiles(mol)
        frags_smiles.append(smiles)
    return frags_smiles
