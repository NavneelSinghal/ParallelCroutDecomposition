CC := gcc
WFLAGS := -Wall -Wextra

all: tests/gen ompver checker mpiver

ompver: ompver.c
	$(CC) -O0 -fopenmp $(WFLAGS) -o $@ $<

mpiver: mpiver.c
	mpicc $(WFLAGS) -o $@ $<

tests/gen: tests/gen.c
	$(CC) $(WFLAGS) -o $@ $<

checker: checker.c
	$(CC) -O3 -fopenmp $(WFLAGS) -o $@ $<

N := 5000
seed := 30
s := 1
avg := 0
strategy := 4
num_threads := 8

clean:
	@rm -f testmatrix
	@rm -f output_*.txt

run: clean ompver tests/gen
	@./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	@time ./ompver $(N) $(N) testmatrix $(num_threads) $(strategy)

mpirun: clean mpiver tests/gen
	@./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	@mpiexec -n $(num_threads) ./mpiver $(N) $(N) testmatrix

mpicheck: clean mpiver tests/gen checker
	@./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	@mpiexec -n $(num_threads) ./mpiver $(N) $(N) testmatrix
	@./checker $(N) $(N) \
	  testmatrix output_L_$(num_threads).txt \
	  output_D_$(num_threads).txt \
	  output_U_$(num_threads).txt

check: clean ompver tests/gen checker
	@./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	@time ./ompver $(N) $(N) testmatrix $(num_threads) $(strategy)
	@./checker $(N) $(N) \
	  testmatrix output_L_$(strategy)_$(num_threads).txt \
	  output_D_$(strategy)_$(num_threads).txt \
	  output_U_$(strategy)_$(num_threads).txt
