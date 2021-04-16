rows=$1
filename=$2
num_threads=$3
strategy=$4

# echo $rows
# echo $filename
# echo $num_threads
# echo $strategy

if [[ strategy -eq 4 ]]
then
	mpiexec -n $num_threads ./mpiver $rows $rows $filename
else
	./ompver $rows $rows $filename $num_threads $strategy
fi
