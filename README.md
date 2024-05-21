# AnoChem

<!--![AnoChem_title_img](assets/anochem-image.png)-->
<img src="./assets/anochem-image.png" width="600px"></img>


## Installation

Clone the github repository of AnoChem.

AnoChem is implemented with `conda` environment. For creation of conda environment,

```bash
conda create -n ${ENV_N} python=3.7 -y
```

Activation of conda environment
```bash
conda activate ${ENV_N}
```

```bash
bash install.sh
```

for primary installation of dependencies

For manual installation, create conda environment with those packages:

- `tensorflow=2.7.0` `rdkit=2022.9.3` `openbabel` `PyFingerprint`
- `scipy` `scikit-learn` `pandas` `numpy`
- `tqdm` `ipython`

Some dependencies are required for for manual installation:

- `PyFingerprint` : Try installation via pip and git as described in the `install.sh`


## How to run

#### General usage

For the activation of environment: `conda activate scaffold`

For the usage of AnoChem, try:

```bash
python main.py -i INPUT_FILE -o OUTPUT_DIRECTORY
```


`INPUT_FILE` should contain SMILES of chemical structures.

`OUTPUT_DIRECTORY` is the location of result output directory, default is `./results`

For the results,
- `OUTPUT_DIRECTORY/anochem_score.csv` : the final score
- `OUTPUT_DIRECTORY/final_report.csv` : detailed subscores


#### Generating images of ECFP4 bits to revise

For the inspection of substructural candidates of SMILES, generating substructural images, try:

`python main.py -i INPUT_FILE -o OUTPUT_DIRECTORY -b NUMBER_OF_IMAGES`

`NUMBER_OF_IMAGES` is the number of images for a SMILES to be generated.

Result images is created at `OUTPUT_DIRECTORY/SMILES_ORDER_NO/~`, in order of priority. Each SMILES is numbered in order as the `INPUT_FILE`, and the `SMILES_ORDER_NO` follows this.


#### Example

For your information, there is a test input file, try:

`python main.py -i input/test.smi -o results -b 5`


## Citation

```bibtex
@article{GU20242116,
title = {AnoChem: Prediction of chemical structural abnormalities based on machine learning models},
journal = {Computational and Structural Biotechnology Journal},
volume = {23},
pages = {2116-2121},
year = {2024},
issn = {2001-0370},
doi = {https://doi.org/10.1016/j.csbj.2024.05.017},
url = {https://www.sciencedirect.com/science/article/pii/S2001037024001636},
author = {Changdai Gu and Woo Dae Jang and Kwang-Seok Oh and Jae Yong Ryu},
keywords = {AnoChem, Drug design, Machine learning, Computational chemistry, Cheminformatics},
abstract = {De novo drug design aims to rationally discover novel and potent compounds while reducing experimental costs during the drug development stage. Despite the numerous generative models that have been developed, few successful cases of drug design utilizing generative models have been reported. One of the most common challenges is designing compounds that are not synthesizable or realistic. Therefore, methods capable of accurately assessing the chemical structures proposed by generative models for drug design are needed. In this study, we present AnoChem, a computational framework based on deep learning designed to assess the likelihood of a generated molecule being real. AnoChem achieves an area under the receiver operating characteristic curve score of 0.900 for distinguishing between real and generated molecules. We utilized AnoChem to evaluate and compare the performances of several generative models, using other metrics, namely SAscore and Fréschet ChemNet distance (FCD). AnoChem demonstrates a strong correlation with these metrics, validating its effectiveness as a reliable tool for assessing generative models. The source code for AnoChem is available at https://github.com/CSB-L/AnoChem.}
```


## Contact

For more information : check [CSB_Lab](https://www.csb-lab.net/)

