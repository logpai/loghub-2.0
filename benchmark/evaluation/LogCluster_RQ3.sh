for complex in 0 1 2 3
    do
        echo "python LogCluster_eval.py -full --complex $complex"
        python LogCluster_eval.py -full --complex $complex
    done


for frequent in 5 -5 10 -10 20 -20
    do
        echo "python LogCluster_eval.py -full --frequent $frequent"
        python LogCluster_eval.py -full --frequent $frequent
    done