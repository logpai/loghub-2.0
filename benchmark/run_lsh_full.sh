# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for technique in LSH
do
    echo ${technique}
    python ${technique}_eval.py -full
done
