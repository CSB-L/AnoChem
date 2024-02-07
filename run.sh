# For ordinary usage
python main.py -i ./input/test.smi  -o ./output_test

# -c, --calc_all : To calculate all the subscores not used as an input feature for ensemble model
# -b INT, --bit_image INT : generation of images for each input SMILES and possible revising targets for the structure.
python main.py -i ./input/test.smi  -o ./output_test_calc_all -c -b 8
