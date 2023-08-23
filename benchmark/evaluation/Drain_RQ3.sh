for complex in 0 1 2 3
    do
        python Drain_eval.py -full --complex $complex
    done

for frequent in 5 -5 10 -10 20 -20
    do
        python Drain_eval.py -full --frequent $frequent
    done

