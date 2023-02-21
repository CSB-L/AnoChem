from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import pickle
import argparse
import scaffoldgraph as sg
#from PyFingerprint.All_Fingerprint import get_fingerprint
from PyFingerprint.fingerprint import get_fingerprint

def argument_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--smiles_input', required=True, help="Input Entrez gene ID")
    parser.add_argument('-s', '--scaffold_flag', required=False, help="Domain change flag", action='store_true')
    
    # TODO-remove here
    parser.add_argument('-c', '--classifier_model', required=False, default=None,
                        help="Classifier model if applicable")
    parser.add_argument('-e', '--ensemble_model', required=False, default=None,
                        help="Ensemble model if applicable")
    parser.add_argument('-y', '--y_label', required=False, default=None,
                        help="Label, if applicable. Accuracy and AUC are returned.")
#     parser.add_argument('--ae_only', required=False, default=False,action='store_true',
#                         help="In case to use AE-based models-only")
    # TODO-until here
    
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
        fps = get_fingerprint(smi, fp_type='standard')
        feat_list.append(fps.to_numpy().reshape(1,-1))
#         bits = to_bits(fps, 1024)
#         feat = []
#         for f in bits:
#             feat.append(int(f))
#         feat_list.append(feat)
#     return np.asarray(feat_list)
    return np.concatenate(feat_list,axis=0)

def calculate_pubchem_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        fps = get_fingerprint(smi, fp_type='pubchem')
        feat_list.append(fps.to_numpy().reshape(1,-1))
#         bits = to_bits(fps, 881)
#         feat = []
#         for f in bits:
#             feat.append(int(f))
#         feat_list.append(feat)
#     return np.asarray(feat_list)
    return np.concatenate(feat_list,axis=0)

def calculate_maccs_fingerprints(smiles_list): 
    feat_list = []
    for smi in smiles_list:
        fps = get_fingerprint(smi, fp_type='maccs')
        feat_list.append(fps.to_numpy().reshape(1,-1))
#         bits = to_bits(fps, 166)
#         feat = []
#         for f in bits:
#             feat.append(int(f))
#         feat_list.append(feat)
#     return np.asarray(feat_list)
    return np.concatenate(feat_list,axis=0)

def get_scaffolds(smi):
    mol = Chem.MolFromSmiles(smi)
    frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False)
    frags_smiles = []
    for mol in frags:
        smiles = Chem.MolToSmiles(mol)
        frags_smiles.append(smiles)
    return frags_smiles


def load_rf(model_f):
    with open(model_f,'rb') as f:
        model = pickle.load(f)
    return model
    