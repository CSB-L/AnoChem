#!/usr/bin/env python3
import os
import warnings
import anochem_predict as pred_funcs
import utils
import datetime
import sub_structure_drawing as sub_dr

warnings.filterwarnings("ignore")
PLF_DIR = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = utils.argument_parser()    
    options = parser.parse_args()
    smiles_input = options.smiles_input
    output_dir = options.output_dir    
    os.makedirs(output_dir, exist_ok=True)
    
    smiles_list = pred_funcs.read_smiles(smiles_input)
    result_df = pred_funcs.predict(
        smiles_list=smiles_list,
        output_dir=output_dir,
        ensb_model_f=os.path.join(PLF_DIR,'models/ensemble/LR_ensemble.best_model.pkl'),
        threshold=0.8,
        _calc_all_=options.calc_all,
    )
    
    if options.bit_image:
        for _idx, _smi in enumerate(smiles_list):
            img_d = sub_dr.draw_sub_imgs(
                smiles=_smi,
                max_fing_n=options.bit_image,
                output_dir=os.path.join(output_dir,str(_idx)),
            )
    
    end = datetime.datetime.now()
    print('Time cost:', end-start)
    