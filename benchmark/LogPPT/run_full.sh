#!/usr/bin/zsh
source ~/.zshrc

conda activate LogPPT
python fewshot_sampling.py
python convert_fewshot_label.py
./train_full.sh
conda deactivate

cd ../evaluation/
conda activate logevaluate
./LogPPT_RQ3.sh
conda deactivate

cd ../LogPPT
