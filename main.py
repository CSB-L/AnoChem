#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
import tqdm
import utils
import os
import glob
import subprocess

warnings.filterwarnings("ignore")
PLF_DIR = os.path.abspath(os.path.dirname(__file__))

def calc_sim(original_feat, pred_feat):
    cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
    return cos_sim

def check_structure_original(smiles, loaded_encoder_model, fingerprint_type, threshold):
    frags = utils.get_scaffolds(smiles)
    if fingerprint_type == 'ecfp':
        feat = utils.calculate_ecfp_fingerprints([smiles])
    elif fingerprint_type == 'maccs':
        feat = utils.calculate_maccs_fingerprints([smiles])
    elif fingerprint_type == 'daylight':
        feat = utils.calculate_daylight_fingerprints([smiles])
    elif fingerprint_type == 'pubchem':
        feat = utils.calculate_pubchem_fingerprints([smiles])
    
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)
    
    sim = calc_sim(feat[0], pred_feat[0])
    return sim


def check_structure_scaffold(smiles, loaded_encoder_model, fingerprint_type, threshold):
    frags = utils.get_scaffolds(smiles)
    if fingerprint_type == 'ecfp':
        feat = utils.calculate_ecfp_fingerprints(frags+[smiles])
    elif fingerprint_type == 'maccs':
        feat = utils.calculate_maccs_fingerprints(frags+[smiles])
    elif fingerprint_type == 'daylight':
        feat = utils.calculate_daylight_fingerprints(frags+[smiles])
    elif fingerprint_type == 'pubchem':
        feat = utils.calculate_pubchem_fingerprints(frags+[smiles])
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)
    
    flag = True
    min_sim = 1.0
    sim_list = []
    for i in range(len(feat)):
        sim = calc_sim(feat[i], pred_feat[i])
        sim_list.append(sim)
        if sim < min_sim:
            min_sim = sim
            
        #if sim < threshold:
        #    flag = False
    #return flag
    return np.min(sim_list)

def read_smiles(smiles_input):
    smiles_list = []
    with open(smiles_input, 'r') as fp:
        for line in fp:
            smiles_list.append(line.strip())
    return smiles_list

# def predict(smiles):
#     model_json = './models/scaffold/256_1_0.001.json'
#     model_weight = './models/scaffold/256_1_0.001.h5'
    
#     json_file = open(model_json, "r")
#     loaded_model_json = json_file.read() 
#     json_file.close()
    
#     loaded_encoder_model = model_from_json(loaded_model_json)
#     loaded_encoder_model.load_weights(model_weight)
    
#     flag = check_structure_scaffold(smiles, loaded_encoder_model)
#     return flag


def run_classification(smiles_input:list,model):
    feats = utils.calculate_ecfp_fingerprints(smiles_input)
    _p = model.predict_proba(feats)
    return _p[:,1].squeeze() # p.shape = (n(feats),label), label 1 is real


# def run_chemprop(output_dir):
#     subprocess.call('chemprop_predict --test_path %s/fingerprint_result.csv --checkpoint_dir models/Scaffold_config_save --preds_path %s/final_result.csv --features_generator rdkit_2d_normalized --no_features_scaling --ensemble_variance'%(output_dir, output_dir), shell=True, stderr=subprocess.STDOUT)
#     return

if __name__ == '__main__':
    parser = utils.argument_parser()    
    options = parser.parse_args()
    smiles_input = options.smiles_input
    output_dir = options.output_dir    
    scaffold_flag = options.scaffold_flag
    threshold = 0.8
    models = {}
    
    os.makedirs(output_dir,exist_ok=True)
    
    # Call anomaly detection models
    if not scaffold_flag:
        folders = glob.glob(os.path.join(PLF_DIR,'./models/original_*'))
        for each_folder in folders:
            basename = os.path.basename(each_folder).split('_')[1].strip()
            model_json = glob.glob(each_folder+'/*.json')[0]
            model_weight = glob.glob(each_folder+'/*.h5')[0]
            
            json_file = open(model_json, "r")
            loaded_model_json = json_file.read() 
            json_file.close()
            loaded_encoder_model = model_from_json(loaded_model_json)
            loaded_encoder_model.load_weights(model_weight)
            models[basename] = loaded_encoder_model
            
        func = check_structure_original
    else:
        folders = glob.glob(os.path.join(PLF_DIR,'./models/scaffold_*'))
        for each_folder in folders:
            basename = os.path.basename(each_folder).split('_')[1].strip()
            model_json = glob.glob(each_folder+'/*.json')[0]
            model_weight = glob.glob(each_folder+'/*.h5')[0]
            
            json_file = open(model_json, "r")
            loaded_model_json = json_file.read() 
            json_file.close()
            loaded_encoder_model = model_from_json(loaded_model_json)
            loaded_encoder_model.load_weights(model_weight)
            models[basename] = loaded_encoder_model
            
        func = check_structure_scaffold
    
    smiles_list = read_smiles(smiles_input)
    
    fp = open(output_dir+'/fingerprint_result.csv', 'w')
    fp.write('%s,%s,%s,%s,%s\n'%('Smiles', 'ECFP4', 'MACCS', 'Daylight', 'PubChem'))
    for smiles in tqdm.tqdm(smiles_list):
        try:
            result_info = {}
            for each_model in models:
                loaded_encoder_model = models[each_model]
                flag = func(smiles, loaded_encoder_model, each_model, threshold)
                result_info[each_model] = flag

            flag_list = []
            for each_model in ['ecfp', 'maccs', 'daylight', 'pubchem']:
                flag_list.append(str(result_info[each_model]))
            fp.write('%s,%s\n'%(smiles, ','.join(flag_list)))
        except:
            continue
            
    fp.close()
    
    # Call classification/ensemble models
    cls_model_f = os.path.join(os.path.split(os.path.abspath(__file__))[0],'models/rfs/cls.pkl')
    cls_model = utils.load_rf(model_f=cls_model_f)
    
    # Classification
    cls_ps = run_classification(smiles_input=smiles_list,model=cls_model)
    cls_report_df = pd.DataFrame([cls_ps],columns=smiles_list,index=['RF_cls_prob']).T
    cls_report_df.index.name='Smiles'
    cls_report_df.to_csv(os.path.join(output_dir,'classficiation_probability.csv'))
    
    # Ensemble
#     ensb_model_f = glob.glob(os.path.join(PLF_DIR,'./models/rfs/ensb.pkl'))
#     ensb_model = utils.load_rf(model_f=ensb_model_f)
    ano_prop_df = pd.read_csv(os.path.join(output_dir,'fingerprint_result.csv'),index_col=0)
    ensb_input_df = pd.concat([ano_prop_df,cls_report_df],axis=1)
#     ensb_ps = ensb_model.predict_proba(np.array(ensb_input_df))[:,1].squeeze()
#     ensb_ps_df = pd.DataFrame([ensb_ps],columns=smiles_list,index='AnoChem_Final_Score').T
#     final_result = pd.concat([ensb_input_df,ensb_ps_df],axis=1)
    
#     final_result.to_csv(os.path.join(output_dir,'results.csv'))
    
#     run_chemprop(output_dir)
    
    # TODO-remove here
    ensb_input_df.to_csv(os.path.join(output_dir,'_ensemble_input.tsv'),sep='\t')
