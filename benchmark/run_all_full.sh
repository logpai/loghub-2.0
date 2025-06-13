# source ~/.zshrc
# conda activate logevaluate

cd evaluation
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell LogPPT UniParser
for technique in AEL Drain LogPPT UniParser IPLoM LenMa LFA LogCluster LogLSHD LogMine Logram LogSig MoLFI SHISO SLCT Spell
do
    echo ${technique}
    python ${technique}_eval.py -full
done
