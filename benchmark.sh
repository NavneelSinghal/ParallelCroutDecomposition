for n in 200 1000 5000
do
    make gen N=$n
    for threads in 1 2 4 8 16
    do
        if [[ threads -ge 4 ]]
        then
            for strategy in 4
            do
                echo N = $n, Threads = $threads, Strategy = $strategy
                time bash run.sh $n testmatrix $threads $strategy
                echo
            done
        else
            for strategy in 4
            do
                echo N = $n, Threads = $threads, Strategy = $strategy
                time bash run.sh $n testmatrix $threads $strategy
                echo
            done
        fi
    done
done
make clean
