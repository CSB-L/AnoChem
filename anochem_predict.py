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
import tempfile


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
    # dropping errored smiles
    _feat_dropped_df = pd.DataFrame(feat,index=smiles_list).dropna()
    _feat_dropped_val = np.array(_feat_dropped_df.values)
    results = loaded_encoder_model.predict(_feat_dropped_val)
    pred_feat = np.where(results > 0.5, 1, 0)
    
    sims = []
    for _ in range(results.shape[0]):
        sims.append(calc_sim(_feat_dropped_val[_,:], pred_feat[_,:]))
    sims_d = pd.Series(sims,index=_feat_dropped_df.index).to_dict()
    return pd.Series(sims_d,index=smiles_list)


def read_smiles(smiles_input):
    smiles_list = []
    with open(smiles_input, 'r') as fp:
        for line in fp:
            smiles_list.append(line.strip())
    return smiles_list


def _run_sklrn_cls(ecfp4_fing:'np.array',model):
    _p = model.predict_proba(ecfp4_fing)
    return _p[:,1].squeeze() # p.shape = (n(feats),label), label 1 is real
    
def _run_tf_cls(ecfp4_fing:'np.array',model):
    _p = model.predict(ecfp4_fing)
    return _p

def get_feats(smiles_input:list):
    feats = utils.calculate_ecfp_fingerprints(smiles_input)
    return feats


def predict(smiles_list:list,
            output_dir:str,
            ensb_model_f:str=os.path.join(PLF_DIR,'models/ensemble/LR_ensemble.best_model.pkl'),
            threshold:float=0.8,
            _calc_all_:bool=False,
           ):
    """
    Main prediction
    smiles_list (list) : list of SMILES, which are str type
    output_dir (str) : location of output directory
    NOTE that any exist files with a identic files would be overwritten
    ensb_model_f (str) : in case to specify an ensemble model file (there is an default model)
    threshold (float) : a threshold for anomaly detection models
    """
    models = {}
    # Call anomaly detection models
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

    # check duplicated smiles
    print("Duplication of input SMILES: ",len(smiles_list)-len(set(smiles_list)))
    smiles_list = list(set(smiles_list))
    
    # Molecular properties
    mol_props_df = utils.get_molecular_prop(smiles_list=smiles_list)
    mol_props_df.to_csv(os.path.join(output_dir,'molecular_properties.csv'))
    
    strc_result = []
    print('Calculating structure-based recovery...')
    if _calc_all_:
        _fingerprints_ = ['ecfp', 'maccs', 'daylight', 'pubchem']
    else:
        _fingerprints_ = ['ecfp']
    
    for each_model in tqdm.tqdm(_fingerprints_):
        _result_ser = func(smiles_list,models[each_model],each_model,threshold)
        _result_ser.name = each_model
        _result_ser.index.name = 'Smiles'
        
        strc_result.append(_result_ser)
    strc_result = pd.concat(strc_result,axis=1)
    if _calc_all_:
        strc_result = strc_result.rename({
            'ecfp':'ECFP4', 'maccs':'MACCS', 'daylight':'Daylight', 'pubchem':'PubChem'},axis=1)
    else:
        strc_result = strc_result.rename({
            'ecfp':'ECFP4'},axis=1)
    strc_result.to_csv(os.path.join(output_dir,'fingerprint_result.csv'))
    
    # Call classification/ensemble models
    print('Under real/generated classification...')
    cls_dnn_model_f = os.path.join(PLF_DIR,'models/classification/cls_dnn.hdf5')
    cls_rf_model_f = os.path.join(PLF_DIR,'models/classification/cls_rf.pkl')
    cls_xgb_model_f = os.path.join(PLF_DIR,'models/classification/XGBoost.best_model.pkl')
    cls_lr_model_f = os.path.join(PLF_DIR,'models/classification/LR.best_model.pkl')
    if not os.path.isfile(cls_xgb_model_f):
        os.system(f'gzip -dk {cls_xgb_model_f}')
    # Call classification models
    xgb_cls_model = utils.load_sklrn_m(model_f=cls_xgb_model_f)
    if _calc_all_:
        dnn_cls_model = utils.load_tf_m(model_f=cls_dnn_model_f)
        rf_cls_model = utils.load_sklrn_m(model_f=cls_rf_model_f)
        lr_cls_model = utils.load_sklrn_m(model_f=cls_lr_model_f)

    # Featurization
    feats = get_feats(smiles_input=smiles_list)
        # null_drop
    feats_df = pd.DataFrame(feats, index=smiles_list)
    feats_null_dropped = feats_df.dropna()
    
    # Classification
    xgb_cls_ps = _run_sklrn_cls(ecfp4_fing=np.array(feats_null_dropped.values),model=xgb_cls_model)
    if _calc_all_:
        dnn_cls_ps = _run_tf_cls(ecfp4_fing=np.array(feats_null_dropped.values),model=dnn_cls_model)
        rf_cls_ps = _run_sklrn_cls(ecfp4_fing=np.array(feats_null_dropped.values),model=rf_cls_model)
        lr_cls_ps = _run_sklrn_cls(ecfp4_fing=np.array(feats_null_dropped.values),model=lr_cls_model)
        
        cls_report_df = pd.DataFrame(
            [
                dnn_cls_ps.reshape(-1,),
                rf_cls_ps.reshape(-1,),
                xgb_cls_ps.reshape(-1,),
                lr_cls_ps.reshape(-1,)
            ],
            columns=feats_null_dropped.index,
            index=['DNN_cls_prob','RF_cls_prob','XGBoost_cls_prob','LR_cls_prob']).T
    else:
        cls_report_df = pd.DataFrame(
            [xgb_cls_ps.reshape(-1,)],
            columns=feats_null_dropped.index,index=['XGBoost_cls_prob']).T
    cls_report_df = pd.DataFrame(cls_report_df,index=smiles_list)
    cls_report_df.index.name='Smiles'
    cls_report_df.to_csv(os.path.join(output_dir,'classficiation_probability.csv'))
    
    ensb_input_df = pd.concat([mol_props_df,strc_result,cls_report_df],axis=1)
    ensb_input_df.to_csv(os.path.join(output_dir,'_ensemble_input.tsv'),sep='\t')
    
    # Dropping null
    ensemble_target_features = [
        'ECFP4',
#         'MACCS','Daylight','PubChem','DNN_cls_prob','RF_cls_prob',
        'XGBoost_cls_prob',
#         'LR_cls_prob',
#         'SA',
        'QED','MW',
#         'LogP',
    ]
    _ensb_input_refined = ensb_input_df.dropna()
    _ensb_input_refined = _ensb_input_refined.loc[:,ensemble_target_features] # with RF_cls_prob
    
    if not os.path.isfile(ensb_model_f):
        if os.path.isfile(ensb_model_f+'.gz'):
            os.system("gzip -dk %s"%ensb_model_f+'.gz')
        else:
            raise(IOError('Cannot find ensemble model'))
            
    ensb_model = utils.load_sklrn_m(model_f=ensb_model_f)
    ensb_ps = ensb_model.predict_proba(np.array(_ensb_input_refined))[:,1].squeeze()
    ensb_ps_df = pd.DataFrame([ensb_ps],columns=_ensb_input_refined.index,index=['AnoChem_Final_Score']).T
    ensb_ps_df.to_csv(os.path.join(output_dir,'anochem_score.csv'))
    
    final_report = pd.concat([ensb_input_df,ensb_ps_df],axis=1)
    final_report.to_csv(os.path.join(output_dir,'final_report.csv'))
    
    return final_report
    

# prediction function for web-version
def _web_post_prediction_(smiles):
    temp_dir = tempfile.TemporaryDirectory()
    output_dir = temp_dir.name
    output_dir = output_dir+'/'
    print('**', output_dir)
    
    results = predict([smiles],output_dir=output_dir)
    results = results.iloc[0,:].to_dict()
    results = {i:f"{j:.4f}" for i,j in results.items()}
    temp_dir.cleanup()
    print ('final_report', results)
    
    return results

