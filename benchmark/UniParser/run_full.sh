#!/usr/bin/zsh
source ~/.zshrc

export CUDA_VISIBLE_DEVICES=2
conda activate UniParser
unset LD_LIBRARY_PATH
python process_log_parsing_input_to_ner.py -full
python TrainNERLogAll.py -full
python InferNERLogAll.py -full
conda deactivate

cd ../evaluation/
conda activate logevaluate
python UniParser_eval.py -full
./UniParser_RQ3.sh

cd ../../UniParser
conda deactivate
