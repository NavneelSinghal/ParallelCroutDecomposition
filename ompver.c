#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int num_threads;

double **alloc_matrix(int n, int m) {
    double **mat = (double **)malloc(sizeof(double *) * n);
#pragma omp parallel for shared(mat, n, m) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        mat[i] = (double *)malloc(sizeof(double) * m);
        double *row = mat[i];
        for (int j = 0; j < m; j++)
            row[j] = 0;
    }
    return mat;
}

void dealloc_matrix(double **a, int n) {
    for (int i = 0; i < n; ++i)
        free(a[i]);
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

void strategy2(double **A, double **L, double **U, int n) {

    // int sqrt_threads = 0;
    // while (sqrt_threads * sqrt_threads <= num_threads)
    //     sqrt_threads++;
    // sqrt_threads--;

    int pw_sqrt = 1;
    while (pw_sqrt * pw_sqrt <= num_threads)
        pw_sqrt <<= 1;
    pw_sqrt >>= 1;

    pw_sqrt = num_threads / pw_sqrt;

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

void strategy3(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

#pragma omp parallel for private(i) shared(A, L, U, n) num_threads(num_threads)
    for (i = 0; i < n; i++)
        U[i][i] = 1;

    for (j = 0; j < n; j++) {
#pragma omp parallel private(i, k, sum) shared(A, L, U, n, j)                  \
    num_threads(num_threads)
        {
#pragma omp sections
            {
#pragma omp section
                {
                    for (i = j + 1; i < n; i++) {
                        sum = 0;
                        for (k = 0; k < j; k++)
                            sum += L[i][k] * U[k][j];
                        L[i][j] = A[i][j] - sum;
                    }
                }
#pragma omp section
                {
                    for (i = j; i <= j; i++) {
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
        }
    }
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

    double **A = alloc_matrix(n, m), **L = alloc_matrix(n, m),
           **U = alloc_matrix(n, m), **D = alloc_matrix(n, m);

    /* Read A from inputfile */
    FILE *input = fopen(inputfile, "r");
    if (input == NULL) {
        fprintf(stderr, "Error while opening input file\n");
        return -1;
    }
    read_matrix(input, A, n, m);
    fclose(input);

    // for time being assuming strategy 1 = serial
    switch (strategy) {
    case 1:
        strategy1(A, L, U, n);
        break;
    case 2:
        strategy2(A, L, U, n);
        break;
    case 3:
        strategy3(A, L, U, n);
        break;
    case 4:
        /* strategy4(A, L, U, n); */
        /* break; */
    default:
        fprintf(stderr, "Strategy %d not recognized/implemented\n", strategy);
    }

    // Construct D matrix
    LtoD(L, D, n, m);

    /* Print L matrix (make it unit) */
    char buffer[1000];
    sprintf(buffer, "output_L_%d_%d.txt", strategy, num_threads);
    FILE *lfile = fopen(buffer, "w");
    print_matrix(lfile, L, n, m);
    fclose(lfile);

    /* Print D matrix */
    sprintf(buffer, "output_D_%d_%d.txt", strategy, num_threads);
    FILE *dfile = fopen(buffer, "w");
    print_matrix(dfile, D, n, m);
    fclose(dfile);

    /* Print U matrix */
    sprintf(buffer, "output_U_%d_%d.txt", strategy, num_threads);
    FILE *ufile = fopen(buffer, "w");
    print_matrix(ufile, U, n, m);
    fclose(ufile);

    /* Deallocate matrices -- change before submission */
    dealloc_matrix(A, n);
    dealloc_matrix(L, n);
    dealloc_matrix(U, n);
    dealloc_matrix(D, n);

    return 0;
}
