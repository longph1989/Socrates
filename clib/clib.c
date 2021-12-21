#include <math.h>
#include <stdlib.h>
#include <stdio.h>


typedef struct layer {
    double *lw;
    double *up;
    double **le;
    double **ge;
} layer;


typedef struct network {
    layer **layers;
} network;


network *create_network(size_t num_layers) {
    network *nn = (network *)malloc(sizeof(network));
    return nn;
}


void add_first_layer(network *nn, double *lw, double *up) {
    layer *new_layer = (layer *)malloc(sizeof(layer));
    
    new_layer->lw = lw;
    new_layer->up = up;
    new_layer->le = NULL;
    new_layer->ge = NULL;

    nn->layers[0] = new_layer;
}


void add_other_layers(network *nn, double *lw, double *up, double **le, double **ge, size_t idx) {
    layer *new_layer = (layer *)malloc(sizeof(layer));
    
    new_layer->lw = lw;
    new_layer->up = up;
    new_layer->le = le;
    new_layer->ge = ge;

    nn->layers[idx] = new_layer;
}


void free_network(network *nn, size_t size) {
    for (int i = 0; i < size; i++) {
        free(nn->layers[i]);
    }
    free(nn);
}


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


double *compute_lower_bounds(double **coefs, double **previous_lower_bounds, double **previous_upper_bounds) {
    return NULL;
}


double *compute_upper_bounds(double **coefs, double **previous_lower_bounds, double **previous_upper_bounds) {
    return NULL;
}
