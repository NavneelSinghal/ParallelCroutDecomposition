CC := gcc
WFLAGS := -Wall -Wextra

all: tests/gen ompver checker

ompver: ompver.c
	$(CC) -O0 -fopenmp $(WFLAGS) -o $@ $<

tests/gen: tests/gen.c
	$(CC) $(WFLAGS) -o $@ $<

checker: checker.c
	$(CC) -O3 -fopenmp $(WFLAGS) -o $@ $<

N := 500
seed := 0
s := 1
avg := 0
strategy := 2
num_threads := 4

clean:
	@rm -f testmatrix
	@rm -f output_*.txt

check: clean ompver tests/gen checker
	@./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	@time ./ompver $(N) $(N) testmatrix $(num_threads) $(strategy)
	@./checker $(N) $(N) \
	  testmatrix output_L_$(strategy)_$(num_threads).txt \
	  output_D_$(strategy)_$(num_threads).txt \
	  output_U_$(strategy)_$(num_threads).txt
