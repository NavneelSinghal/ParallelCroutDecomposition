#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int num_threads;
double start_time, end_time;

#define TIMEIT_START (start_time = omp_get_wtime())
#define TIMEIT_END(section)                                                    \
    end_time = omp_get_wtime();                                                \
    printf(section " time elapsed = %.2lf ms\n", (end_time - start_time) * 1000)

#define min(a, b) a < b ? a : b

double **alloc_matrix(int n, int m) {
    double **mat = (double **)malloc(sizeof(double *) * n);
    double *mem = (double *)malloc(sizeof(double) * n * m);
#pragma omp parallel for shared(mat, n, m) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        mat[i] = &mem[m * i];
        double *row = mat[i];
        for (int j = 0; j < m; j++)
            row[j] = 0;
    }
    return mat;
}

void dealloc_matrix(double **a, int n) {
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

void multiply_matrix(double **A, double **B, double **C, int n) {
#pragma omp parallel for shared(A, B, C, n)
    for (int i = 0; i < n; i++) {
        double *row = C[i];
        for (int j = 0; j < n; j++) {
            double res = 0;
            for (int k = 0; k < n; k++)
                res += A[i][k] * B[k][j];
            row[j] = res;
        }
    }
}

void transpose_matrix(double **U, int n) {
#pragma omp parallel for shared(U, n) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double temp = U[i][j];
            U[i][j] = U[j][i];
            U[j][i] = temp;
        }
    }
}

// long double minabs = 1e18;

void LtoD(double **L, double **D, int n, int m) {
    for (int i = 0; i < n; i++) {
        D[i][i] = L[i][i];
        assert(D[i][i] != 0);
        // if (minabs > abs(D[i][i])) minabs = abs(D[i][i]);
    }
    // printf("minabs = %0.100Lf\n", minabs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            L[i][j] /= D[j][j];
}

void strategy1(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

    for (i = 0; i < n; i++)
        U[i][i] = 1;

    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++)
                sum += L[i][k] * U[k][j];
            L[i][j] = A[i][j] - sum;
        }

        for (i = j; i < n; i++) {
            sum = 0;
            for (k = 0; k < j; k++)
                sum += L[j][k] * U[k][i];
            if (L[j][j] == 0)
                exit(0);
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
}

// check what is the optimal value of pw_sqrt, and if strategy 2x is better than strategy 2

void strategy2(double **A, double **L, double **U, int n) {

    // int sqrt_threads = 0;
    // while (sqrt_threads * sqrt_threads <= num_threads)
    //     sqrt_threads++;
    // sqrt_threads--;

    // int pw_sqrt = 1;
    // while (pw_sqrt * pw_sqrt <= num_threads)
    //     pw_sqrt <<= 1;
    // pw_sqrt >>= 1;
    // pw_sqrt = num_threads / pw_sqrt;
    
    // int pw_sqrt = num_threads / 2;
    int pw_sqrt = 2;

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
#pragma omp parallel for num_threads(pw_sqrt)
        for (int i = j; i < n; ++i) {
            double sum = 0;
#pragma omp parallel num_threads(num_threads / pw_sqrt)
            {
                double s = 0;
#pragma omp for
                for (int k = 0; k < j; ++k)
                    s += L[i][k] * U[k][j];
#pragma omp critical
                sum += s;
            }
            L[i][j] = A[i][j] - sum;
        }

        if (L[j][j] == 0)
            exit(0);

#pragma omp parallel for num_threads(pw_sqrt)
        for (int i = j; i < n; ++i) {
            double sum = 0;
#pragma omp parallel num_threads(num_threads / pw_sqrt)
            {
                double s = 0;
#pragma omp for
                for (int k = 0; k < j; ++k)
                    s += L[j][k] * U[k][i];
#pragma omp critical
                sum += s;
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
}

void strategy21(double **A, double **L, double **U, int n) {

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            double sum = 0;
#pragma omp parallel num_threads(num_threads)
            {
                double s = 0;
#pragma omp for
                for (int k = 0; k < j; ++k)
                    s += L[i][k] * U[k][j];
#pragma omp critical
                sum += s;
            }
            L[i][j] = A[i][j] - sum;
        }

        if (L[j][j] == 0)
            exit(0);

        for (int i = j; i < n; ++i) {
            double sum = 0;
#pragma omp parallel num_threads(num_threads)
            {
                double s = 0;
#pragma omp for
                for (int k = 0; k < j; ++k)
                    s += L[j][k] * U[k][i];
#pragma omp critical
                sum += s;
            }
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
}

void strategy22(double **A, double **L, double **U, int n) {

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
#pragma omp parallel for num_threads(num_threads)
        for (int i = j; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[i][k] * U[k][j];
            L[i][j] = A[i][j] - sum;
        }

        if (L[j][j] == 0)
            exit(0);

#pragma omp parallel for num_threads(num_threads)
        for (int i = j; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[j][k] * U[k][i];
            U[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }
}

void strategy22_transpose(double **A, double **L, double **U, int n) {

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
#pragma omp parallel for num_threads(num_threads)
        for (int i = j; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[i][k] * U[j][k];
            L[i][j] = A[i][j] - sum;
        }

        if (L[j][j] == 0)
            exit(0);

#pragma omp parallel for num_threads(num_threads)
        for (int i = j; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[j][k] * U[i][k];
            U[i][j] = (A[j][i] - sum) / L[j][j];
        }
    }

    transpose_matrix(U, n);
}

void strategy22_pack(double **A, double **L, double **U, int n) {

    // for (int i = 0; i < n; ++i)
    //    U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
#pragma omp parallel for num_threads(num_threads)
        for (int i = j; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[i][k] * L[k][j];
            L[i][j] = A[i][j] - sum;
        }

        if (L[j][j] == 0)
            exit(0);

#pragma omp parallel for num_threads(num_threads)
        for (int i = j + 1; i < n; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[j][k] * L[k][i];
            L[j][i] = (A[j][i] - sum) / L[j][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        U[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            U[i][j] = L[i][j];
            L[i][j] = 0;
        }
    }
}

void strategy23(double **A, double **L, double **U, int n) {

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i <= j; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[i][k] * U[k][j];
            L[i][j] = A[i][j] - sum;
        }
#pragma omp parallel shared(A, L, U, n, j) num_threads(num_threads)
        {
#pragma omp for nowait
            for (int i = j + 1; i < n; ++i) {
                double sum = 0;
                for (int k = 0; k < j; ++k)
                    sum += L[i][k] * U[k][j];
                L[i][j] = A[i][j] - sum;
            }

#pragma omp for
            for (int i = j; i < n; ++i) {
                if (L[j][j] == 0)
                    exit(0);
                double sum = 0;
                for (int k = 0; k < j; ++k)
                    sum += L[j][k] * U[k][i];
                U[j][i] = (A[j][i] - sum) / L[j][j];
            }
        }
    }
}

void strategy24(double **A, double **L, double **U, int n) {

    for (int i = 0; i < n; ++i)
        U[i][i] = 1;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i <= j; ++i) {
            double sum = 0;
            for (int k = 0; k < j; ++k)
                sum += L[i][k] * U[j][k];
            L[i][j] = A[i][j] - sum;
        }
#pragma omp parallel shared(A, L, U, n, j) num_threads(num_threads)
        {
#pragma omp for nowait
            for (int i = j + 1; i < n; ++i) {
                double sum = 0;
                for (int k = 0; k < j; ++k)
                    sum += L[i][k] * U[j][k];
                L[i][j] = A[i][j] - sum;
            }

#pragma omp for
            for (int i = j; i < n; ++i) {
                if (L[j][j] == 0)
                    exit(0);
                double sum = 0;
                for (int k = 0; k < j; ++k)
                    sum += L[j][k] * U[i][k];
                U[i][j] = (A[j][i] - sum) / L[j][j];
            }
        }
    }

    transpose_matrix(U, n);
}

void strategy3(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

#define LLoop(st, en)                                                          \
    for (i = st; i < en; i++) {                                                \
        sum = 0;                                                               \
        for (k = 0; k < j; k++)                                                \
            sum += L[i][k] * U[j][k];                                          \
        L[i][j] = A[i][j] - sum;                                               \
    }

#define ULoop(st, en)                                                          \
    for (i = st; i < en; i++) {                                                \
        sum = 0;                                                               \
        for (k = 0; k < j; k++)                                                \
            sum += L[j][k] * U[i][k];                                          \
        if (L[j][j] == 0)                                                      \
            exit(0);                                                           \
        U[i][j] = (A[j][i] - sum) / L[j][j];                                   \
    }

    for (i = 0; i < n; i++)
        U[i][i] = 1;

    int n2 = n / 2;

    for (j = 0; j < n; j++) {
        LLoop(j, j + 1)
#pragma omp parallel private(i, k, sum) shared(A, L, U, n, j)                  \
    num_threads(num_threads)
        {
#pragma omp sections
            {
#pragma omp section
                { LLoop(j + 1, n2) }
#pragma omp section
                { LLoop(n2, n) }
#pragma omp section
                { ULoop(j, n2) }
#pragma omp section
                { ULoop(n2, n) }
            }
        }
    }

    transpose_matrix(U, n);

#undef LLoop
#undef ULoop
}

void strategy32(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

#define LLoop(st, en)                                                          \
    for (i = st; i < en; i++) {                                                \
        sum = 0;                                                               \
        for (k = 0; k < j; k++)                                                \
            sum += L[i][k] * U[j][k];                                          \
        L[i][j] = A[i][j] - sum;                                               \
    }

#define ULoop(st, en)                                                          \
    for (i = st; i < en; i++) {                                                \
        sum = 0;                                                               \
        for (k = 0; k < j; k++)                                                \
            sum += L[j][k] * U[i][k];                                          \
        if (L[j][j] == 0)                                                      \
            exit(0);                                                           \
        U[i][j] = (A[j][i] - sum) / L[j][j];                                   \
    }

    for (i = 0; i < n; i++)
        U[i][i] = 1;

    for (j = 0; j < n; j++) {
        LLoop(j, j + 1)
#pragma omp parallel private(i, k, sum) shared(A, L, U, n, j)                  \
    num_threads(num_threads)
        {
#pragma omp sections
            {
#pragma omp section
                { LLoop(j + 1, n) }
#pragma omp section
                { ULoop(j, n) }
            }
        }
    }

    transpose_matrix(U, n);

#undef LLoop
#undef ULoop
}

void strategy4(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

    for (i = 0; i < n; i++)
        U[i][i] = 1;

    for (j = 0; j < n; j++) {
#pragma omp parallel private(i, k, sum) shared(A, L, U, n, j)                  \
    num_threads(min(2, num_threads))
        {
#pragma omp sections
            {
#pragma omp section
                {
#pragma omp parallel for private(i, k, sum) shared(A, L, U, n, j)              \
    num_threads((num_threads + 1) / 2)
                    for (i = j + 1; i < n; i++) {
                        sum = 0;
                        for (k = 0; k < j; k++)
                            sum += L[i][k] * U[j][k];
                        L[i][j] = A[i][j] - sum;
                    }
                }

#pragma omp section
                {
                    for (i = j; i <= j; i++) {
                        sum = 0;
                        for (k = 0; k < j; k++)
                            sum += L[i][k] * U[j][k];
                        L[i][j] = A[i][j] - sum;
                    }
#pragma omp parallel for private(i, k, sum) shared(A, L, U, n, j)              \
    num_threads((num_threads + 1) / 2)
                    for (i = j; i < n; i++) {
                        sum = 0;
                        for (k = 0; k < j; k++)
                            sum += L[j][k] * U[i][k];
                        if (L[j][j] == 0)
                            exit(0);
                        U[i][j] = (A[j][i] - sum) / L[j][j];
                    }
                }
            }
        }
    }

    transpose_matrix(U, n);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Usage: %s N M inputfile num_threads strategy\n", argv[0]);
        return -1;
    }

    int n = atoi(argv[1]), m = atoi(argv[2]);
    assert(n == m);

    char *inputfile = argv[3];
    num_threads = atoi(argv[4]);
    int strategy = atoi(argv[5]);
    if (num_threads > 1) {
        omp_set_nested(1);
    } else {
        omp_set_nested(0);
    }

    TIMEIT_START;
    double **A = alloc_matrix(n, m), **L = alloc_matrix(n, m),
           **U = alloc_matrix(n, m), **D = alloc_matrix(n, m);
    TIMEIT_END("malloc");

    /* Read A from inputfile */
    TIMEIT_START;
    FILE *input = fopen(inputfile, "r");
    if (input == NULL) {
        fprintf(stderr, "Error while opening input file\n");
        return -1;
    }
    read_matrix(input, A, n, m);
    fclose(input);
    TIMEIT_END("Reading matrix");

    // for time being assuming strategy 1 = serial
    TIMEIT_START;
    switch (strategy) {
    case 1:
        strategy1(A, L, U, n);
        break;
    case 2:
        // strategy2(A, L, U, n);
        // strategy21(A, L, U, n);
        // strategy22(A, L, U, n);
        // strategy22_pack(A, L, U, n);
        strategy22_transpose(A, L, U, n);
        // strategy23(A, L, U, n);
        // strategy24(A, L, U, n);
        break;
    case 3:
        /* strategy3(A, L, U, n); */
        strategy32(A, L, U, n);
        break;
    case 4:
        strategy4(A, L, U, n);
        break;
    default:
        fprintf(stderr, "Strategy %d not recognized/implemented\n", strategy);
    }
    TIMEIT_END("Decomposition");

    TIMEIT_START;
    // Construct D matrix
    LtoD(L, D, n, m);
    TIMEIT_END("D matrix");

    TIMEIT_START;
#pragma omp parallel shared(L, D, U, n, m, strategy, num_threads)              \
    num_threads(num_threads)
    {
#pragma omp sections
        {
#pragma omp section
            {
                /* Print L matrix (make it unit) */
                char buffer[1000];
                sprintf(buffer, "output_L_%d_%d.txt", strategy, num_threads);
                FILE *lfile = fopen(buffer, "w");
                print_matrix(lfile, L, n, m);
                fclose(lfile);
            }
#pragma omp section
            {
                /* Print D matrix */
                char buffer[1000];
                sprintf(buffer, "output_D_%d_%d.txt", strategy, num_threads);
                FILE *dfile = fopen(buffer, "w");
                print_matrix(dfile, D, n, m);
                fclose(dfile);
            }
#pragma omp section
            {
                /* Print U matrix */
                char buffer[1000];
                sprintf(buffer, "output_U_%d_%d.txt", strategy, num_threads);
                FILE *ufile = fopen(buffer, "w");
                print_matrix(ufile, U, n, m);
                fclose(ufile);
            }
        }
    }
    TIMEIT_END("Printing");

    /* Deallocate matrices -- change before submission */
    TIMEIT_START;
    dealloc_matrix(A, n);
    dealloc_matrix(L, n);
    dealloc_matrix(U, n);
    dealloc_matrix(D, n);
    TIMEIT_END("Free");

    return 0;
}
