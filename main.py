import warnings
import numpy as np
from tensorflow.keras.models import model_from_json 
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
import tqdm
import utils
import os
import glob
warnings.filterwarnings("ignore")

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
    if sim >= threshold:
        return True
    else:
        return False

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
    for i in range(len(feat)):
        sim = calc_sim(feat[i], pred_feat[i])
        if sim < threshold:
            flag = False
    return flag

def read_smiles(smiles_input):
    smiles_list = []
    with open(smiles_input, 'r') as fp:
        for line in fp:
            smiles_list.append(line.strip())
    return smiles_list

def predict(smiles):
    model_json = './models/scaffold/256_1_0.001.json'
    model_weight = './models/scaffold/256_1_0.001.h5'
    
    json_file = open(model_json, "r")
    loaded_model_json = json_file.read() 
    json_file.close()
    
    loaded_encoder_model = model_from_json(loaded_model_json)
    loaded_encoder_model.load_weights(model_weight)
    
    flag = check_structure_scaffold(smiles, loaded_encoder_model)
    return flag

if __name__ == '__main__':
    parser = utils.argument_parser()    
    options = parser.parse_args()
    smiles_input = options.smiles_input
    output_file = options.output_dir    
    scaffold_flag = options.scaffold_flag
    threshold = 0.8
    models = {}
    
    if not scaffold_flag:
        folders = glob.glob('./models/original_*')
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
        folders = glob.glob('./models/scaffold_*')
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
    
    fp = open(output_file, 'w')
    fp.write('%s\t%s\t%s\t%s\t%s\n'%('Smiles', 'ECFP4', 'MACCS', 'Daylight', 'PubChem'))
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
        except:
            continue
        fp.write('%s\t%s\n'%(smiles, '\t'.join(flag_list)))    
    fp.close()
