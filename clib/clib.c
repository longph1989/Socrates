#include "clib.h"


network *create_network(const size_t num_layers) {
    network *nn = (network *) malloc(sizeof(network));
    nn->layers = (layer **) malloc(num_layers * sizeof(layer *));
    nn->num_layers = num_layers;

    return nn;
}


void add_first_layer(const network *nn, double *lw, double *up, const size_t num_neurons) {
    layer *new_layer = (layer *) malloc(sizeof(layer));
    
    new_layer->lw = lw;
    new_layer->up = up;
    new_layer->le = NULL;
    new_layer->ge = NULL;
    new_layer->num_neurons = num_neurons;

    nn->layers[0] = new_layer;
}


void add_other_layers(const network *nn, double *lw, double *up, double **le, double **ge, const size_t num_neurons, const size_t idx) {
    layer *new_layer = (layer *) malloc(sizeof(layer));
    
    new_layer->lw = lw;
    new_layer->up = up;
    new_layer->le = le;
    new_layer->ge = ge;
    new_layer->num_neurons = num_neurons;

    nn->layers[idx] = new_layer;
}


void free_network(network *nn, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}


void *compute_lower_bounds_thread(void *args) {
    thread_arg_t *data = (thread_arg_t *) args;

    network *nn = data->nn;
    size_t start = data->start;
    size_t end = data->end;
    size_t row = data->row;
    size_t col = data->col;
    double **coefs = data->coefs;
    double *res = data->res;

    for (size_t i = start; i < end; i++) {
        double res_i = 0.0;
        double *coefs_i = coefs[i];
        double *coefs_next = NULL;

        for (int j = nn->num_layers - 1; j >= 0; j--) {
            layer *layer_j = nn->layers[j];
            double *lw_j = layer_j->lw;
            double *up_j = layer_j->up;
            double **le_j = layer_j->le;
            double **ge_j = layer_j->ge;

            if (j == 0) {
                res_i = 0.0;

                for (size_t k = 0; k < layer_j->num_neurons; k++) {
                    if (coefs_i[k] > 0) {
                        res_i += coefs_i[k] * lw_j[k];
                    } else if (coefs_i[k] < 0) {
                        res_i += coefs_i[k] * up_j[k];
                    }
                }

                res_i += coefs_i[layer_j->num_neurons];
            }
            else {
                layer *layer_j1 = nn->layers[j - 1];

                res_i = 0.0;
                coefs_next = (double *) malloc((layer_j1->num_neurons + 1) * sizeof(double));

                for (size_t k = 0; k < layer_j1->num_neurons + 1; k++) {
                    coefs_next[k] = 0.0;
                }

                for (size_t k = 0; k < layer_j->num_neurons; k++) {
                    if (coefs_i[k] > 0) {
                        res_i += coefs_i[k] * lw_j[k];
                        compute_coefs_next(coefs_next, coefs_i[k], ge_j[k], layer_j1->num_neurons + 1);
                        // coefs_next += coefs_i[k] * ge_j[k]; //
                    } else if (coefs_i[k] < 0) {
                        res_i += coefs_i[k] * up_j[k];
                        compute_coefs_next(coefs_next, coefs_i[k], le_j[k], layer_j1->num_neurons + 1);
                        // coefs_next += coefs_i[k] * le_j[k]; //
                    }
                }

                res_i += coefs_i[layer_j->num_neurons];
                coefs_next[layer_j1->num_neurons] += coefs_i[layer_j->num_neurons];
            }

            if (j == nn->num_layers - 1) {
                res[i] = res_i;
                coefs_i = coefs_next;
            } else {
                res[i] = MAX(res[i], res_i);
                free_array(coefs_i);
                coefs_i = coefs_next;
            }
        }
    }

    return NULL;
}


void *compute_upper_bounds_thread(void *args) {
    thread_arg_t *data = (thread_arg_t *) args;

    network *nn = data->nn;
    size_t start = data->start;
    size_t end = data->end;
    size_t row = data->row;
    size_t col = data->col;
    double **coefs = data->coefs;
    double *res = data->res;

    for (size_t i = start; i < end; i++) {
        double res_i = 0.0;
        double *coefs_i = coefs[i];
        double *coefs_next = NULL;

        for (int j = nn->num_layers - 1; j >= 0; j--) {
            layer *layer_j = nn->layers[j];
            double *lw_j = layer_j->lw;
            double *up_j = layer_j->up;
            double **le_j = layer_j->le;
            double **ge_j = layer_j->ge;

            if (j == 0) {
                res_i = 0.0;

                for (size_t k = 0; k < layer_j->num_neurons; k++) {
                    if (coefs_i[k] > 0) {
                        res_i += coefs_i[k] * up_j[k];
                    } else if (coefs_i[k] < 0) {
                        res_i += coefs_i[k] * lw_j[k];
                    }
                }

                res_i += coefs_i[layer_j->num_neurons];
            } else {
                layer *layer_j1 = nn->layers[j - 1];

                res_i = 0.0;
                coefs_next = (double *) malloc((layer_j1->num_neurons + 1) * sizeof(double));

                for (size_t k = 0; k < layer_j1->num_neurons + 1; k++) {
                    coefs_next[k] = 0.0;
                }

                for (size_t k = 0; k < layer_j->num_neurons; k++) {
                    if (coefs_i[k] > 0) {
                        res_i += coefs_i[k] * up_j[k];
                        compute_coefs_next(coefs_next, coefs_i[k], le_j[k], layer_j1->num_neurons + 1);
                        // coefs_next += coefs_i[k] * le_j[k]; //
                    } else if (coefs_i[k] < 0) {
                        res_i += coefs_i[k] * lw_j[k];
                        compute_coefs_next(coefs_next, coefs_i[k], ge_j[k], layer_j1->num_neurons + 1);
                        // coefs_next += coefs_i[k] * ge_j[k]; //
                    }
                }

                res_i += coefs_i[layer_j->num_neurons];
                coefs_next[layer_j1->num_neurons] += coefs_i[layer_j->num_neurons];
            }

            if (j == nn->num_layers - 1) {
                res[i] = res_i;
                coefs_i = coefs_next;
            } else {
                res[i] = MIN(res[i], res_i);
                free_array(coefs_i);
                coefs_i = coefs_next;
            }
        }
    }

    return NULL;
}


double *compute_lower_bounds(network *nn, double **coefs, const size_t row, const size_t col, const size_t NUM_THREADS) {
    double *res = (double *) malloc(row * sizeof(double));
    for (size_t i = 0; i < row; i++)
        res[i] = 0.0;

    pthread_t threads[NUM_THREADS];
    thread_arg_t thread_args[NUM_THREADS];

    size_t idx_start = 0;
    size_t idx_n = row / NUM_THREADS;
    size_t idx_end = idx_start + idx_n;

    size_t i = 0;
    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i].nn = nn;
        thread_args[i].start = idx_start;
        thread_args[i].end = idx_end;
        thread_args[i].coefs = coefs;
        thread_args[i].row = row;
        thread_args[i].col = col;
        thread_args[i].res = res; 

        idx_start = idx_end;
        idx_end = idx_start + idx_n;

        if (i == NUM_THREADS - 2) {
            idx_end = row;
        }

        pthread_create(&threads[i], NULL, compute_lower_bounds_thread, (void *) &thread_args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return res;
}


double *compute_upper_bounds(network *nn, double **coefs, const size_t row, const size_t col, const size_t NUM_THREADS) {
    double *res = (double *) malloc(row * sizeof(double));
    for (size_t i = 0; i < row; i++)
        res[i] = 0.0;

    pthread_t threads[NUM_THREADS];
    thread_arg_t thread_args[NUM_THREADS];

    size_t idx_start = 0;
    size_t idx_n = row / NUM_THREADS;
    size_t idx_end = idx_start + idx_n;

    size_t i = 0;
    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i].nn = nn;
        thread_args[i].start = idx_start;
        thread_args[i].end = idx_end;
        thread_args[i].coefs = coefs;
        thread_args[i].row = row;
        thread_args[i].col = col;
        thread_args[i].res = res; 

        idx_start = idx_end;
        idx_end = idx_start + idx_n;

        if (i == NUM_THREADS - 2) {
            idx_end = row;
        }

        pthread_create(&threads[i], NULL, compute_upper_bounds_thread, (void *) &thread_args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return res;
}


void free_array(double *array) {
    free(array);
}


void compute_coefs_next(double *coefs_next, const double coef, const double *values, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        coefs_next[i] += coef * values[i];
    }
}


double *array_mul_c(double* matrix, double *coefs, int *indexes, int m, int n) {
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
