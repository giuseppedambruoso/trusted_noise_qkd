#!/bin/bash
# wrapper.sh

N_VAL=$1
P_VAL=$2
ALPHA_VAL=$3

# Attiva il virtualenv
source /lustrehome/giosca/SCALAG/trusted_noise2/qkd_venv/bin/activate

# Assicurati che pip --user sia nel PATH
export PATH=$PATH:$(python3 -m site --user-base)/bin

# Crea cartella dei risultati e lancia il wrapper
python main.py $N_VAL $P_VAL $ALPHA_VAL --output_dir results
