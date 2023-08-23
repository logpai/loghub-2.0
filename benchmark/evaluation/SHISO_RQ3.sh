for complex in 0 1 2 3
    do
        python SHISO_eval.py -full --complex $complex
    done


for frequent in 5 -5 10 -10 20 -20
    do
        python SHISO_eval.py -full --frequent $frequent
    done