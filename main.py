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
import datetime

warnings.filterwarnings("ignore")
PLF_DIR = os.path.abspath(os.path.dirname(__file__))

def calc_sim(original_feat, pred_feat):
#     cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
    cos_sim = np.min([dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat)),1.0])
    return cos_sim

def check_structure_original(smiles_list:list,
                             loaded_encoder_model,
                             fingerprint_type:str,
                             threshold:float
                            )->'pd.Series':
    if fingerprint_type == 'ecfp':
        feat = utils.calculate_ecfp_fingerprints(smiles_list)
    elif fingerprint_type == 'maccs':
        feat = utils.calculate_maccs_fingerprints(smiles_list)
    elif fingerprint_type == 'daylight':
        feat = utils.calculate_daylight_fingerprints(smiles_list)
    elif fingerprint_type == 'pubchem':
        feat = utils.calculate_pubchem_fingerprints(smiles_list)
    
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)
    
    sims = []
    for _ in range(results.shape[0]):
        sims.append(calc_sim(feat[_,:], pred_feat[_,:]))
    
#     sim = calc_sim(feat[0], pred_feat[0])
    return pd.Series(sims,index=smiles_list)


def check_structure_scaffold(smiles_list:list,
                             loaded_encoder_model,
                             fingerprint_type:str,
                             threshold:float,
                            )->'pd.Series':
    smi_to_frag = {_smi:utils.get_scaffolds(_smi) for _smi in smiles_list} #{SMILES:FRAG_LIST}
    smi_frags = []
    smi_idx = {}
    curr_idx = 0
    for _smi in smiles_list:
        frags = smi_to_frag[_smi]
        smi_frags.append(_smi)
        smi_frags.extend(frags)
        
        _frag_n = len(frags)
        _next_idx = curr_idx+_frag_n+1
        smi_idx[_smi] = (curr_idx,_next_idx)
        curr_idx = _next_idx

    if fingerprint_type == 'ecfp':
        feat = utils.calculate_ecfp_fingerprints(smi_frags)
    elif fingerprint_type == 'maccs':
        feat = utils.calculate_maccs_fingerprints(smi_frags)
    elif fingerprint_type == 'daylight':
        feat = utils.calculate_daylight_fingerprints(smi_frags)
    elif fingerprint_type == 'pubchem':
        feat = utils.calculate_pubchem_fingerprints(smi_frags)
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)
    
#     flag = True
#     min_sim = 1.0
    sims = {}
    for _smi in smiles_list:
        sim_list = [calc_sim(feat[k,:], pred_feat[k,:]) for k in range(smi_idx[_smi][0],smi_idx[_smi][1],1)]
#         for i in range(len(feat)):
#             sim = calc_sim(feat[i], pred_feat[i])
#             sim_list.append(sim)
#             if sim < min_sim:
#                 min_sim = sim

            #if sim < threshold:
            #    flag = False
        #return flag
        sims[_smi] = np.min(sim_list)
#     return np.min(sim_list)
    return pd.Series(sims,index=smiles_list)


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


def _run_rf_cls(ecfp4_fing:'np.array',model):
    _p = model.predict_proba(ecfp4_fing)
    return _p[:,1].squeeze() # p.shape = (n(feats),label), label 1 is real
    
def _run_dnn_cls(ecfp4_fing:'np.array',model):
    _p = model.predict(ecfp4_fing)
    return _p

def get_feats(smiles_input:list):
    feats = utils.calculate_ecfp_fingerprints(smiles_input)
    return feats


# def run_chemprop(output_dir):
#     subprocess.call('chemprop_predict --test_path %s/fingerprint_result.csv --checkpoint_dir models/Scaffold_config_save --preds_path %s/final_result.csv --features_generator rdkit_2d_normalized --no_features_scaling --ensemble_variance'%(output_dir, output_dir), shell=True, stderr=subprocess.STDOUT)
#     return

if __name__ == '__main__':
    start = datetime.datetime.now()
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
    
    strc_result = []
    print('Calculating structure-based recovery...')
    for each_model in tqdm.tqdm(['ecfp', 'maccs', 'daylight', 'pubchem']):
        _result_ser = func(smiles_list,models[each_model],each_model,threshold)
        _result_ser.name = each_model
        _result_ser.index.name = 'Smiles'
        
        strc_result.append(_result_ser)
    strc_result = pd.concat(strc_result,axis=1)
    strc_result = strc_result.rename({
        'ecfp':'ECFP4', 'maccs':'MACCS', 'daylight':'Daylight', 'pubchem':'PubChem'},axis=1)
    strc_result.to_csv(output_dir+'/fingerprint_result.csv')
    
#     fp = open(output_dir+'/fingerprint_result.csv', 'w')
#     fp.write('%s,%s,%s,%s,%s\n'%('Smiles', 'ECFP4', 'MACCS', 'Daylight', 'PubChem'))
#     fp_err = open(os.path.join(output_dir,'fingerprint_errored.tsv'),'w')
#     fp_err.write('Smiles\tError\n')
#     for smiles in tqdm.tqdm(smiles_list):
#         try:
#             result_info = {}
#             for each_model in models:
# #                 loaded_encoder_model = models[each_model]
# #                 flag = func(smiles, loaded_encoder_model, each_model, threshold)
#                 flag = func(smiles, models[each_model], each_model, threshold)
#                 result_info[each_model] = flag

#             flag_list = []
#             for each_model in ['ecfp', 'maccs', 'daylight', 'pubchem']:
#                 flag_list.append(str(result_info[each_model]))
#             fp.write('%s,%s\n'%(smiles, ','.join(flag_list)))
#         except Exception as e:
#             fp_err.write(f'{smiles}\t{e}\n')
#             continue
            
#     fp.close()
#     fp_err.close()
    
    # Call classification/ensemble models
    print('Under real/generated classification...')
    cls_dnn_model_f = os.path.join(PLF_DIR,'models/classification/cls_dnn.hdf5')
    cls_rf_model_f = os.path.join(PLF_DIR,'models/classification/cls_rf.pkl')
    
#     if options.classifier_model:
#         if os.path.isfile(options.classifier_model):
#             rf_cls_model = options.classifier_model
    # TODO-to here
    dnn_cls_model = utils.load_dnn(model_f=cls_dnn_model_f)
    rf_cls_model = utils.load_rf(model_f=cls_rf_model_f)

    # Featurization
    feats = get_feats(smiles_input=smiles_list)
    
    # Classification
    dnn_cls_ps = _run_dnn_cls(ecfp4_fing=feats,model=dnn_cls_model)
    rf_cls_ps = _run_rf_cls(ecfp4_fing=feats,model=rf_cls_model)
    
    cls_report_df = pd.DataFrame([dnn_cls_ps.reshape(-1,),rf_cls_ps.reshape(-1,)],columns=smiles_list,index=['DNN_cls_prob','RF_cls_prob']).T
    cls_report_df.index.name='Smiles'
    cls_report_df.to_csv(os.path.join(output_dir,'classficiation_probability.csv'))
    
    # Ensemble
    ensb_input_df = pd.concat([strc_result,cls_report_df],axis=1)
    # TODO-remove here
    ensb_input_df.to_csv(os.path.join(output_dir,'_ensemble_input.tsv'),sep='\t')
    
#     # Dropping null
#     ensb_input_nadroped = ensb_input_df.dropna()
    
#     ensb_model_f = glob.glob(os.path.join(PLF_DIR,'./models/ensemble/ensb.pkl'))
#     if options.ensemble_model:
#         if os.path.isfile(options.ensemble_model):
#             ensb_model_f = options.ensemble_model
#             ensb_model = utils.load_rf(model_f=ensb_model_f)
#             ensb_ps = ensb_model.predict_proba(np.array(ensb_input_nadroped))[:,1].squeeze()
#             ensb_ps_df = pd.DataFrame([ensb_ps],columns=smiles_list,index=['AnoChem_Final_Score']).T
#             ensb_ps_df.to_csv(os.path.join(output_dir,'final_score.csv'))
#             final_report = pd.concat([ensb_input_df,ensb_ps_df],axis=1)
            
            
#             final_report.to_csv(os.path.join(output_dir,'final_report.csv'))
            

#             # TODO - remove here : calculation of model performance
#             if options.y_label: # .npy file
#                 if os.path.isfile(options.y_label):
#                     import calc_metrics
#                     import json
#                     y_true = np.load(options.y_label)
#                     # if null dropped
# #                     if set(ensb_input_df.index) != set(ensb_input_nadroped.index):
#                     valid_idx = [i in ensb_input_nadroped.index for i in ensb_input_df.index]
#                     y_true = y_true[valid_idx]
#                     assert y_true.shape == y_pred.shape
#                     perform_dict = calc_metrics.calc_model_performance(
#                         y_true=y_true,y_pred=ensb_ps)
#                     with open(os.path.join(output_dir,'AnoChem_Performance.json'),'wb') as f:
#                         f.write(json.dumps(perform_dict).encode())
            
    end = datetime.datetime.now()
    print('Time cost:', end-start)
    