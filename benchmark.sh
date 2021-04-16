for n in 5000
do
    make gen N=$n
    for threads in 1 2 4 8 16
    do
        for strategy in 0 1 2 3 4
        do
            echo N = $n, Threads = $threads, Strategy = $strategy
            time bash run.sh $n testmatrix $threads $strategy
            echo
        done
    done
done
make clean
