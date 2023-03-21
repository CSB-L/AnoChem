#!/usr/bin/env python3
import os
import warnings
import anochem_predict as predict
import utils
import datetime

warnings.filterwarnings("ignore")
PLF_DIR = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = utils.argument_parser()    
    options = parser.parse_args()
    smiles_input = options.smiles_input
    output_dir = options.output_dir    
    os.makedirs(output_dir, exist_ok=True)
    
    smiles_list = predict.read_smiles(smiles_input)
    result_df = predict.predict(smiles_list=smiles_list,
                                output_dir=output_dir,
                                ensb_model_f=os.path.join(PLF_DIR,'models/ensemble/ensemble_model.LR.pkl'),
                                ensemble_with_RF=True,
                                threshold=0.8,
                                save_tmp_reports=False,
                               )
    
    end = datetime.datetime.now()
    print('Time cost:', end-start)
    