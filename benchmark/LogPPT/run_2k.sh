#!/usr/bin/zsh
source ~/.zshrc

conda activate LogPPT
export CUDA_VISIBLE_DEVICES=4

python fewshot_sampling.py
./train_2k.sh

cd ../evaluation/
conda activate logevaluate
python LogPPT_eval.py -otc

cd ../LogPPT
