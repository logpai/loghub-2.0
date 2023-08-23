# source ~/.zshrc
# conda activate logevaluate

cd evaluation
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell
# for technique in LogSig MoLFI SHISO SLCT Spell
# for technique in LogSig
for technique in Drain
do
    echo ${technique}
    python ${technique}_eval.py -otc
    python ${technique}_eval.py -full
done
