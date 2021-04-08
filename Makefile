CC := gcc
WFLAGS := -Wall -Wextra

all: tests/gen ompver

ompver: ompver.c
	$(CC) -O0 -fopenmp $(WFLAGS) -o $@ $<

tests/gen: tests/gen.c
	gcc $(WFLAGS) -o $@ $<

N := 100
seed := 0
s := 1
avg := 0
strategy := 1
num_threads := 4

clean:
	rm -rf testmatrix temp/
	rm -f output_*.txt

check: clean ompver tests/gen
	./tests/gen $(N) $(seed) $(s) $(avg) > testmatrix
	./ompver $(N) $(N) testmatrix $(num_threads) 1
	@mkdir temp/
	@mv output_L_1_$(num_threads).txt \
	   output_D_1_$(num_threads).txt \
	   output_U_1_$(num_threads).txt \
	   temp/
	time ./ompver $(N) $(N) testmatrix $(num_threads) $(strategy)
	@diff -s temp/output_L_1_$(num_threads).txt output_L_$(strategy)_$(num_threads).txt
	@diff -s temp/output_D_1_$(num_threads).txt output_D_$(strategy)_$(num_threads).txt
	@diff -s temp/output_U_1_$(num_threads).txt output_U_$(strategy)_$(num_threads).txt
