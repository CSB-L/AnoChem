import warnings
import numpy as np
from tensorflow.keras.models import model_from_json 
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
import tqdm
import utils
import os

warnings.filterwarnings("ignore")

def calc_sim(original_feat, pred_feat):
    cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
    return cos_sim

def check_structure_original(smiles, loaded_encoder_model):
    feat = utils.calculate_fingerprints([smiles])
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)

    sim = calc_sim(feat[0], pred_feat[0])
    if sim > 0.9:        
        return True
    else:
        return False

def check_structure_scaffold(smiles, loaded_encoder_model):
    frags = utils.get_scaffolds(smiles)
    feat = utils.calculate_fingerprints(frags+[smiles])
    
    results = loaded_encoder_model.predict(feat)
    pred_feat = np.where(results > 0.5, 1, 0)
    
    flag = True
    for i in range(len(feat)):
        sim = calc_sim(feat[i], pred_feat[i])
        if sim < 0.9:
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
    
    if not scaffold_flag:
        model_json = './models/original/256_1_0.001.json'
        model_weight = './models/original/256_1_0.001.h5'
        func = check_structure_original
    else:
        model_json = './models/scaffold/256_1_0.001.json'
        model_weight = './models/scaffold/256_1_0.001.h5'
        func = check_structure_scaffold
    
    json_file = open(model_json, "r")
    loaded_model_json = json_file.read() 
    json_file.close()
    
    loaded_encoder_model = model_from_json(loaded_model_json)
    loaded_encoder_model.load_weights(model_weight)
    
    smiles_list = read_smiles(smiles_input)
    
    fp = open(output_file, 'w')
    for smiles in tqdm.tqdm(smiles_list):
        flag = func(smiles, loaded_encoder_model)
        fp.write('%s\t%s\n'%(smiles, flag))
    fp.close()

    # cos_sim = dot(original_feat, pred_feat)/(norm(original_feat)*norm(pred_feat))
    
#     # smiles_list = []
#     # with open('../2.feature_calculation/results_ECFP4_test/smi_pairs.txt', 'r') as fp:
#     #     for line in fp:
#     #         sptlist = line.strip().split('\t')
#     #         smiles_list.append([sptlist[0].strip(), sptlist[1].strip()])                           

#     val_x = np.load(x_file)
#     val_x = val_x[0:100]
#     print (val_x.shape)




#     fp = open('./result_test.txt', 'w')
#     for i in range(len(results)):
#         original_feat = val_x[i]
#         pred_feat = results[i]
#         smiles = 'N/A'#smiles_list[i][1]
#         dist = np.linalg.norm(original_feat-pred_feat)    
#         dist = distance.jaccard(original_feat, pred_feat)
#         
#         fp.write('%s\t%s\t%s\n'%(smiles, cos_sim, dist))
#     fp.close()