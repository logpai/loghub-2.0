# source ~/.zshrc
# conda activate logevaluate

cd evaluation
for technique in AEL Drain IPLoM LenMa LFA LogCluster LogLSHD LogMine Logram LogSig MoLFI SHISO SLCT Spell
do
    echo ${technique}
    python ${technique}_eval.py -full
done
