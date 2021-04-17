size=$1
filename=$2

bash compile.sh
for strategy in {0..4}
do
  for num_threads in 1 2 4 8 16
  do
    echo Now checking strategy $strategy @ $num_threads threads!
    bash run.sh $size $filename $num_threads $strategy
    python3 format_checker.py $filename output_L_${strategy}_${num_threads}.txt \
      output_U_${strategy}_${num_threads}.txt
  done
done
