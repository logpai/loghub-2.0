# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for technique in LogLSHD
do
    echo ${technique}
    python ${technique}_eval.py -full
done
