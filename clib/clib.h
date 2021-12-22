#ifndef CLIB_H
#define CLIB_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "macros.h"


typedef struct layer {
    double *lw;
    double *up;
    double **le;
    double **ge;
    size_t num_neurons;
} layer;

typedef struct network {
    layer **layers;
    size_t num_layers;
} network;

typedef struct thread_arg_t {
    network *nn;
    size_t start;
    size_t end;
    double **coefs;
    size_t row;
    size_t col;
    double *res;
} thread_arg_t;


network *create_network(const size_t num_layers);

void add_first_layer(const network *nn, double *lw, double *up, const size_t num_neurons);

void add_other_layers(const network *nn, double *lw, double *up, double **le, double **ge, const size_t num_neurons, const size_t idx);

void free_network(network *nn, const size_t size);

void *compute_lower_bounds_thread(void *args);

void *compute_upper_bounds_thread(void *args);

double *compute_lower_bounds(network *nn, double **coefs, const size_t row, const size_t col, const size_t NUM_THREADS);

double *compute_upper_bounds(network *nn, double **coefs, const size_t row, const size_t col, const size_t NUM_THREADS);

void free_array(double *b);

void compute_coefs_next(double *coefs_next, const double coef, const double *values, const size_t size);


#endif /* CLIB_H */
