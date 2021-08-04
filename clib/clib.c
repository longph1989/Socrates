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

    printf("\ncoefs\n");

    for (int i = 0; i < m; i++) {
        int index = indexes[i];
        printf("%.12f, ", coefs[index]);
    }

    printf("\n");

    for (int i = 0; i < m; i++) {
        int index = indexes[i];

        if (i == 0) {
            printf("\nvalues\n");
        }

        for (int j = 0; j < n; j++) {
            double value = *(matrix + index * n + j);
            if (i == 0) {
                if (value != 0) {
                    printf("%.12f, ", value);
                }
            }
            if (value != 0) {
                res[j] += value * coefs[index];
            }
        }

    }

    printf("\n");

    return res;
}
