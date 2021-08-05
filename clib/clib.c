#include <math.h>
#include <stdlib.h>
#include <stdio.h>


void free_array(double *b) {
    // printf("free\n");
    free(b);
}

double *array_mul_c(double* matrix, double* coefs, int* indexes, int m, int n) {
    double *res = malloc(n * sizeof(double));

    for (int j = 0; j < n; j++) {
        res[j] = 0;
    }

    for (int i = 0; i < m; i++) {
        int index = indexes[i];

        for (int j = 0; j < n; j++) {
            double value = *(matrix + index * n + j);
            if (value != 0) {
                res[j] += value * coefs[index];
            }
        }

    }

    return res;
}
