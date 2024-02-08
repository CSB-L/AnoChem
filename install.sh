#ENV_N="anochem"
#conda create -n ${ENV_N} python=3.7 -y && \
#conda activate ${ENV_N} && \
pip install git+https://github.com/hcji/PyFingerprint.git && \
conda env update -f environment.yml -n ${ENV_N} --prune


