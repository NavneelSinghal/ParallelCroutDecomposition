#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int rank, num_processes;
double start_time, end_time;

#define TIMEIT_START                                                           \
    if (rank == 0) {                                                           \
        start_time = MPI_Wtime();                                              \
    }
#define TIMEIT_END(section)                                                    \
    if (rank == 0) {                                                           \
        end_time = MPI_Wtime();                                                \
        printf(section " time elapsed = %.2lf ms\n",                           \
               (end_time - start_time) * 1000);                                \
    }

int min(int a, int b) { return a < b ? a : b; }

double **alloc_matrix(int n, int m) {
    double **mat = (double **)malloc(sizeof(double *) * n);
    double *mem = (double *)malloc(sizeof(double) * n * m);
    for (int i = 0; i < n; i++) {
        mat[i] = &mem[m * i];
        double *row = mat[i];
        for (int j = 0; j < m; j++)
            row[j] = 0;
    }
    return mat;
}

void dealloc_matrix(double **a) {
    free(*a);
    free(a);
}

void read_matrix(FILE *input, double **mat, int n, int m) {
    double num;
    for (int i = 0; i < n; i++) {
        double *row = mat[i];
        for (int j = 0; j < m; j++) {
            fscanf(input, "%lf", &num);
            row[j] = num;
        }
    }
}

void print_matrix(FILE *output, double **mat, int n, int m) {
    for (int i = 0; i < n; i++) {
        double *row = mat[i];
        for (int j = 0; j < m; j++)
            fprintf(output, "%.9lf ", row[j]);
        fprintf(output, "\n");
    }
}

void transpose_matrix(double **U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double temp = U[i][j];
            U[i][j] = U[j][i];
            U[j][i] = temp;
        }
    }
}

void LtoD(double **L, double **D, int n, int m) {
    for (int i = 0; i < n; i++) {
        D[i][i] = L[i][i];
        assert(D[i][i] != 0);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            L[i][j] /= D[j][j];
}

void crout(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

    /* Let each worker do this part O(n) */
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    /* Allocate a buffer of size 2*n for communication */
    /* Stack should be sufficient (16kB for n~1024) */
    double buffer[2 * n];

    for (j = 0; j < n; j++) {
        /* Let each worker compute L[j][j] on its own O(n) */
        for (i = j; i <= j; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            L[i][j] = A[i][j] - sum;
        }

        /* We now need to iterate from j+1 -> n in num_processes chunks.
         * Verify that :
         *      st[rank=0] = 0
         *      en[rank=n-1] = n
         *      en[rank=i] = st[rank=i+1]
         */
        int st = (j + 1) + ((n - (j + 1)) / num_processes) * rank +
                 min((n - (j + 1)) % num_processes, rank);
        int en = (j + 1) + ((n - (j + 1)) / num_processes) * (rank + 1) +
                 min((n - (j + 1)) % num_processes, rank + 1);

        for (i = st; i < en; i++) {
            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[i][k] * U[k][j];
            }
            /* L[i][j] = A[i][j] - sum; */
            buffer[2 * (i - st)] = A[i][j] - sum;

            sum = 0;
            for (k = 0; k < j; k++) {
                sum = sum + L[j][k] * U[k][i];
            }
            if (L[j][j] == 0) {
                fprintf(stderr, "Fatal: Non-decomposable\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            buffer[2 * (i - st) + 1] = (A[j][i] - sum) / L[j][j];
        }

        /* Gather all results of this iteration into master */
        if (rank != 0)
            MPI_Gather(buffer, 2 * (en - st), MPI_DOUBLE, NULL, 2 * (en - st),
                       MPI_DOUBLE, 0, MPI_COMM_WORLD);
        else
            MPI_Gather(MPI_IN_PLACE, 2 * (en - st), MPI_DOUBLE, buffer,
                       2 * (en - st), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Broadcast buffer from master back to all workers */
        MPI_Bcast(buffer, 2 * (n - (j + 1)), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Copy buffer into respective matrices */
        for (i = j + 1; i < n; i++) {
            L[i][j] = buffer[2 * (i - (j + 1))];
            U[j][i] = buffer[2 * (i - (j + 1)) + 1];
        }

        /* Brace for next iteration */
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

        TIMEIT_START;

    if (argc < 4) {
        if (rank == 0) {
            // Only one process should print error messages
            printf("Usage: %s N M inputfile\n", argv[0]);
        }
        // But all processes with terminate
        return -1;
    }

    int n = atoi(argv[1]), m = atoi(argv[2]);
    assert(n == m);

    char *inputfile = argv[3];

    double **A = alloc_matrix(n, m), **L = alloc_matrix(n, m),
           **U = alloc_matrix(n, m), **D;
    if (rank == 0) {
        D = alloc_matrix(n, m);
    }

    if (rank == 0) {
        /* Read A from inputfile (only master process) */
        FILE *input = fopen(inputfile, "r");
        if (input == NULL) {
            fprintf(stderr, "Error while opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        read_matrix(input, A, n, m);
        fclose(input);
        TIMEIT_END("initialization");
    }

    /* Real code starts now */

    // Step 1 : Broadcast matrix A to all workers
    TIMEIT_START;
    MPI_Bcast(*A, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    TIMEIT_END("Matrix broadcast");

    // Step 2 : Decompose matrix in parallel
    TIMEIT_START;
    crout(A, L, U, n);
    TIMEIT_END("Decomposition");

    /* All processes other than master can exit */
    if (rank == 0) {
        TIMEIT_START;
        // Construct D matrix
        LtoD(L, D, n, m);
        TIMEIT_END("D matrix");

        TIMEIT_START;
        /* Print L matrix (make it unit) */
        char buffer[1000];
        sprintf(buffer, "output_L_%d.txt", num_processes);
        FILE *lfile = fopen(buffer, "w");
        print_matrix(lfile, L, n, m);
        fclose(lfile);
        /* Print D matrix */
        sprintf(buffer, "output_D_%d.txt", num_processes);
        FILE *dfile = fopen(buffer, "w");
        print_matrix(dfile, D, n, m);
        fclose(dfile);
        /* Print U matrix */
        sprintf(buffer, "output_U_%d.txt", num_processes);
        FILE *ufile = fopen(buffer, "w");
        print_matrix(ufile, U, n, m);
        fclose(ufile);
        TIMEIT_END("Printing");
    }
    dealloc_matrix(A);
    dealloc_matrix(L);
    dealloc_matrix(U);
    if (rank == 0)
        dealloc_matrix(D);
    MPI_Finalize();
    return 0;
}