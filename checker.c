#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-3

double dabs(double a) { return a < 0 ? -a : a; }

double **alloc_matrix(int n, int m) {
    double **mat = (double **)malloc(sizeof(double *) * n);
#pragma omp parallel for shared(mat, n, m)
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

int main(int argc, char **argv) {
    assert(argc == 7);
    int n = atoi(argv[1]), m = atoi(argv[2]);

    double **A = alloc_matrix(n, m), **L = alloc_matrix(n, m),
           **D = alloc_matrix(n, m), **U = alloc_matrix(n, m),
           **X = alloc_matrix(n, m), **Y = alloc_matrix(n, m);

    // read matrices
    FILE *afile = fopen(argv[3], "r"), *lfile = fopen(argv[4], "r"),
         *dfile = fopen(argv[5], "r"), *ufile = fopen(argv[6], "r");
    read_matrix(afile, A, n, m);
    read_matrix(lfile, L, n, m);
    read_matrix(dfile, D, n, m);
    read_matrix(ufile, U, n, m);

    // multiply matrices
    multiply_matrix(L, D, X, n);
    multiply_matrix(X, U, Y, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (dabs(A[i][j] - Y[i][j]) > EPSILON) {
                fprintf(
                    stderr,
                    "A != LDU (match fail at row=%d col=%d : %lf != %lf) \n", i,
                    j, A[i][j], Y[i][j]);
                return 0;
            }

    dealloc_matrix(A, n);
    dealloc_matrix(L, n);
    dealloc_matrix(D, n);
    dealloc_matrix(U, n);
    dealloc_matrix(X, n);
    dealloc_matrix(Y, n);

    printf("Verified A = L D U : pass\n");
}
