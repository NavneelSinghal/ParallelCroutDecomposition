#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

    int seed = 0;
    int n = 1;
    double s = 1.0;
    double avg = 0.0;

    if (argc <= 1) {
        printf("usage: %s <n> <seed> <s> <avg> "
               "to generate random square matrices of size n, with scale s and "
               "mean avg\n",
               argv[0]);
        exit(0);
    }

    if (argc >= 5)
        avg = atof(argv[4]);
    if (argc >= 4)
        s = atof(argv[3]);
    if (argc >= 3)
        seed = atoi(argv[2]);
    if (argc >= 2)
        n = atoi(argv[1]);

    srand(seed);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%0.12f ", s * (2 * rand() - RAND_MAX) / RAND_MAX + avg);
        printf("\n");
    }

    return 0;
}
