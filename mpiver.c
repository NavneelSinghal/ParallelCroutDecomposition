#include <assert.h>
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

int rank, num_processes;
double start_time, end_time;

#define RAW_TIMEIT_START start_time = MPI_Wtime();
#define RAW_TIMEIT_END(section)                                                \
    {                                                                          \
        end_time = MPI_Wtime();                                                \
        printf(section " time elapsed = %.2lf ms\n",                           \
               (end_time - start_time) * 1000);                                \
    }

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

void write_output(char fname[], double **arr, int n) {
    FILE *f = fopen(fname, "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%0.12f ", arr[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
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

#define chunk_size(start, end)                                                 \
    (((end) - (start) + (num_processes)-1) / (num_processes))

void crout(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

    /* Let each worker do this part O(n) */
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    /* Allocate a buffer of size 2*n for communication */
    /* Stack should be sufficient (16kB for n~1024) */
    double *buffer =
        (double *)malloc(2 * chunk_size(0, n) * num_processes * sizeof(double));

    for (j = 0; j < n; j++) {
        /* Let each worker compute L[j][j] on its own O(n) */
        sum = 0;
        for (k = 0; k < j; k++)
            sum += L[j][k] * U[k][j];
        L[j][j] = A[j][j] - sum;
        if (L[j][j] == 0) {
            fprintf(stderr, "Fatal: Non-decomposable\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* We now need to iterate from j+1 -> n in num_processes chunks.
         * Verify that :
         *      st[rank=0] = 0
         *      en[rank=n-1] = n
         *      en[rank=i] = st[rank=i+1]
         */

        if (j == n - 1) {
            // we probably don't need to compute the rest of it
            break;
        }

        // chunk size becomes 0 if j == n - 1,
        // since there are really no iterations to be done
        int size = chunk_size(j + 1, n);
        int st = j + 1 + rank * size;
        int en = min(n, st + size);

        for (i = st; i < en; i++) {
            sum = 0;
            for (k = 0; k < j; k++)
                sum += L[i][k] * U[k][j];
            /* L[i][j] = A[i][j] - sum; */
            buffer[2 * (i - st)] = A[i][j] - sum;

            sum = 0;
            for (k = 0; k < j; k++)
                sum += L[j][k] * U[k][i];
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            buffer[2 * (i - st) + 1] = (A[j][i] - sum) / L[j][j];
        }

        /* Gather all results of this iteration into master */
        if (rank != 0)
            MPI_Gather(buffer, 2 * size, MPI_DOUBLE, NULL, 2 * size, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);
        else
            MPI_Gather(MPI_IN_PLACE, 2 * size, MPI_DOUBLE, buffer, 2 * size,
                       MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Broadcast buffer from master back to all workers */
        MPI_Bcast(buffer, 2 * size * num_processes, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

        /* Copy buffer into respective matrices */
        for (i = j + 1; i < n; i++) {
            L[i][j] = buffer[2 * (i - (j + 1))];
            U[j][i] = buffer[2 * (i - (j + 1)) + 1];
        }

        /* Brace for next iteration */
    }
    free(buffer);
}

void crout_transpose(double **A, double **L, double **U, int n) {
    int i, j, k;
    double sum = 0;

    /* Let each worker do this part O(n) */
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    /* Allocate a buffer of size 2*n for communication */
    /* Stack should be sufficient (16kB for n~1024) */
    double *buffer =
        (double *)malloc(sizeof(double) * 2 * chunk_size(0, n) * num_processes);

    for (j = 0; j < n; j++) {

        double *Lj = L[j];
        double *Uj = U[j];

        /* Let each worker compute L[j][j] on its own O(n) */
        sum = 0;
        for (k = 0; k < j; k++)
            sum += Lj[k] * Uj[k];
        Lj[j] = A[j][j] - sum;
        if (Lj[j] == 0) {
            fprintf(stderr, "Fatal: Non-decomposable\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* We now need to iterate from j+1 -> n in num_processes chunks.
         * Verify that :
         *      st[rank=0] = 0
         *      en[rank=n-1] = n
         *      en[rank=i] = st[rank=i+1]
         */

        if (j == n - 1) {
            // we probably don't need to compute the rest of it
            break;
        }

        // chunk size becomes 0 if j == n - 1,
        // since there are really no iterations to be done
        int size = chunk_size(j + 1, n);
        int st = j + 1 + rank * size;
        int en = min(n, st + size);

        for (i = st; i < en; i++) {

            double *Li = L[i];
            double *Ui = U[i];

            sum = 0;
            for (k = 0; k < j; k++)
                sum += Li[k] * Uj[k];
            /* L[i][j] = A[i][j] - sum; */
            buffer[2 * (i - st)] = A[i][j] - sum;

            sum = 0;
            for (k = 0; k < j; k++)
                sum += Lj[k] * Ui[k];
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            buffer[2 * (i - st) + 1] = (A[j][i] - sum) / L[j][j];
        }

        /* Gather all results of this iteration into master */
        if (rank != 0)
            MPI_Gather(buffer, 2 * size, MPI_DOUBLE, NULL, 2 * size, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);
        else
            MPI_Gather(MPI_IN_PLACE, 2 * size, MPI_DOUBLE, buffer, 2 * size,
                       MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Broadcast buffer from master back to all workers */
        MPI_Bcast(buffer, 2 * size * num_processes, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

        /* Copy buffer into respective matrices */
        for (i = j + 1; i < n; i++) {
            L[i][j] = buffer[2 * (i - (j + 1))];
            U[i][j] = buffer[2 * (i - (j + 1)) + 1];
        }

        /* Brace for next iteration */
    }

    if (rank == 0)
        transpose_matrix(U, n);

    free(buffer);
}

void crout_gatherall(double **A, double **L, double **U, int n) {

    int i, j, k;
    double sum = 0;

    /* Let each worker do this part O(n) */
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    /* Allocate a buffer of size 2*n for communication */
    /* Stack should be sufficient (16kB for n~1024) */
    double *buffer =
        (double *)malloc(sizeof(double) * 2 * chunk_size(0, n) * num_processes);
    double *recv_buffer =
        (double *)malloc(sizeof(double) * 2 * chunk_size(0, n) * num_processes);

    for (j = 0; j < n; j++) {

        double *Lj = L[j];
        double *Uj = U[j];

        /* Let each worker compute L[j][j] on its own O(n) */
        sum = 0;
        for (k = 0; k < j; k++)
            sum += Lj[k] * Uj[k];
        Lj[j] = A[j][j] - sum;
        if (Lj[j] == 0) {
            fprintf(stderr, "Fatal: Non-decomposable\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* We now need to iterate from j+1 -> n in num_processes chunks.
         * Verify that :
         *      st[rank=0] = 0
         *      en[rank=n-1] = n
         *      en[rank=i] = st[rank=i+1]
         */

        if (j == n - 1) {
            // we probably don't need to compute the rest of it
            break;
        }

        // chunk size becomes 0 if j == n - 1,
        // since there are really no iterations to be done
        int size = chunk_size(j + 1, n);
        int st = j + 1 + rank * size;
        int en = min(n, st + size);

        for (i = st; i < en; i++) {

            double *Li = L[i];
            double *Ui = U[i];

            sum = 0;
            for (k = 0; k < j; k++)
                sum += Li[k] * Uj[k];
            /* L[i][j] = A[i][j] - sum; */
            buffer[2 * (i - st)] = A[i][j] - sum;

            sum = 0;
            for (k = 0; k < j; k++)
                sum += Lj[k] * Ui[k];
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            buffer[2 * (i - st) + 1] = (A[j][i] - sum) / L[j][j];
        }

        /* Gather all results of this iteration into master */
        MPI_Allgather(buffer, 2 * size, MPI_DOUBLE, recv_buffer, 2 * size,
                      MPI_DOUBLE, MPI_COMM_WORLD);

        /* Copy buffer into respective matrices */
        for (i = j + 1; i < n; i++) {
            L[i][j] = recv_buffer[2 * (i - (j + 1))];
            U[i][j] = recv_buffer[2 * (i - (j + 1)) + 1];
        }

        /* Brace for next iteration */
    }

    if (rank == 0)
        transpose_matrix(U, n);
    free(buffer);
    free(recv_buffer);
}

void crout_async(double **A, double **L, double **U, int n) {

    int i, j, k;
    double sum = 0;

    /* Let each worker do this part O(n) */
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    /* Allocate a buffers of size 2*n for communication */
    /* Stack should be sufficient (64kB for n~1024) */
    double **send_buffer = (double **)malloc(sizeof(double *) * 2);
    double **recv_buffer = (double **)malloc(sizeof(double *) * 2);
    for (int i = 0; i < 2; ++i) {
        send_buffer[i] = (double *)malloc(sizeof(double) * 2 *
                                          chunk_size(0, n) * num_processes);
        recv_buffer[i] = (double *)malloc(sizeof(double) * 2 *
                                          chunk_size(0, n) * num_processes);
    }
    int turn = 0; // Which buffer to use this iteration
    MPI_Request allgather_request = MPI_REQUEST_NULL;

    for (j = 0; j < n; j++) {

        double *Lj = L[j];
        double *Uj = U[j];

        double *send_turn = send_buffer[turn];
        double *recv_turn = recv_buffer[turn];

        /* Let each worker compute L[j][j] on its own O(n) */
        sum = 0;
        for (k = 0; k < j - 1; k++)
            sum += Lj[k] * Uj[k];
        Lj[j] = A[j][j] - sum;

        int size = chunk_size(j + 1, n);
        int st = j + 1 + rank * size;
        int en = min(n, st + size);

        for (i = st; i < en; i++) {
            double *Li = L[i];
            double *Ui = U[i];

            sum = 0;
            for (k = 0; k < j - 1; k++)
                sum += Li[k] * Uj[k];
            /* L[i][j] = A[i][j] - sum; */
            send_turn[2 * (i - st)] = A[i][j] - sum;

            sum = 0;
            for (k = 0; k < j - 1; k++)
                sum += Lj[k] * Ui[k];
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            send_turn[2 * (i - st) + 1] = (A[j][i] - sum);
        }

        if (allgather_request != MPI_REQUEST_NULL) {
            int flag = 0;
            while (!flag) {
                MPI_Test(&allgather_request, &flag, MPI_STATUS_IGNORE);
                sched_yield();
            }
            /* Copy buffer into respective matrices */
            for (i = j; i < n; i++) {
                L[i][j - 1] = recv_turn[2 * (i - j)];
                U[i][j - 1] = recv_turn[2 * (i - j) + 1];
            }
        }

        sum = 0;
        for (k = j - 1; k < j; k++)
            sum += Lj[k] * Uj[k];
        Lj[j] -= sum;
        if (Lj[j] == 0) {
            fprintf(stderr, "Fatal: Non-decomposable\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        if (j == n - 1) {
            // we probably don't need to compute the rest of it
            break;
        }

        for (i = st; i < en; i++) {
            double *Li = L[i];
            double *Ui = U[i];

            sum = 0;
            for (k = j - 1; k < j; k++)
                sum += Li[k] * Uj[k];
            /* L[i][j] = A[i][j] - sum; */
            send_turn[2 * (i - st)] -= sum;

            sum = 0;
            for (k = j - 1; k < j; k++)
                sum += Lj[k] * Ui[k];
            /* U[j][i] = (A[j][i] - sum) / L[j][j]; */
            send_turn[2 * (i - st) + 1] =
                (send_turn[2 * (i - st) + 1] - sum) / L[j][j];
        }

        /* Gather all results of this iteration into master */
        MPI_Iallgather(send_turn, 2 * size, MPI_DOUBLE, recv_buffer[turn ^ 1],
                       2 * size, MPI_DOUBLE, MPI_COMM_WORLD,
                       &allgather_request);

        /* Brace for next iteration */
        turn ^= 1;
    }

    if (rank == 0)
        transpose_matrix(U, n);
    for (int i = 0; i < 2; ++i) {
        free(send_buffer[i]);
        free(recv_buffer[i]);
    }
    free(send_buffer);
    free(recv_buffer);
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
           **U = alloc_matrix(n, m);

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
    // crout(A, L, U, n);
    /* crout_transpose(A, L, U, n); */
    /* crout_gatherall(A, L, U, n); */
    crout_async(A, L, U, n);
    TIMEIT_END("Decomposition");

    if (rank == 0) {
        char buffer[1000];
        {
            /* Print L matrix (make it unit) */
            sprintf(buffer, "output_L_4_%d.txt", num_processes);
            write_output(buffer, L, n);
            // FILE *lfile = fopen(buffer, "w");
            // print_matrix(lfile, L, n, m);
            // fclose(lfile);
        }
        {
            /* Print U matrix */
            sprintf(buffer, "output_U_4_%d.txt", num_processes);
            write_output(buffer, U, n);
            // FILE *ufile = fopen(buffer, "w");
            // print_matrix(ufile, U, n, m);
            // fclose(ufile);
        }
    }

    // if (rank == 1 || num_processes == 1) {
    //     /* Print L matrix (make it unit) */
    //     char buffer[1000];
    //     sprintf(buffer, "output_L_%d.txt", num_processes);
    //     FILE *lfile = fopen(buffer, "w");
    //     print_matrix(lfile, L, n, m);
    //     fclose(lfile);
    // }
    // if (rank == 0) {
    //     /* Print U matrix */
    //     char buffer[1000];
    //     sprintf(buffer, "output_U_%d.txt", num_processes);
    //     FILE *ufile = fopen(buffer, "w");
    //     print_matrix(ufile, U, n, m);
    //     fclose(ufile);
    // }
    dealloc_matrix(A);
    dealloc_matrix(L);
    dealloc_matrix(U);

    /* All processes other than master can exit */
    MPI_Finalize();

    return 0;
}
