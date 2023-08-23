#!/usr/bin/zsh
source ~/.zshrc

export CUDA_VISIBLE_DEVICES=1
conda activate UniParser
unset LD_LIBRARY_PATH
python process_log_parsing_input_to_ner.py
python TrainNERLogAll.py
python InferNERLogAll.py
conda deactivate

cd ../evaluation/
conda activate logevaluate
python UniParser_eval.py -otc

cd ../../UniParser
conda deactivate
