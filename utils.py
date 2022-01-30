from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import argparse
import scaffoldgraph as sg
from PyFingerprint.All_Fingerprint import get_fingerprint

def argument_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--smiles_input', required=True, help="Input Entrez gene ID")
    parser.add_argument('-s', '--scaffold_flag', required=False, help="Domain change flag", action='store_true')
    return parser

def to_bits(data, bitlen):
    bit_list = [0]*bitlen
    for val in data:
        bit_list[val] = 1
    return bit_list

def calculate_ecfp_fingerprints(smiles_list): 
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

def calculate_daylight_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        fps = get_fingerprint(smi, fp_type='daylight')
        bits = to_bits(fps, 1024)
        feat = []
        for f in bits:
            feat.append(int(f))
        feat_list.append(feat)
    return np.asarray(feat_list)

def calculate_pubchem_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        fps = get_fingerprint(smi, fp_type='pubchem')
        bits = to_bits(fps, 881)
        feat = []
        for f in bits:
            feat.append(int(f))
        feat_list.append(feat)
    return np.asarray(feat_list)

def calculate_maccs_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        fps = get_fingerprint(smi, fp_type='maccs')
        bits = to_bits(fps, 166)
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
