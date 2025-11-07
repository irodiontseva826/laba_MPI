#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "utils.h"

void process_error(int err){
    if (err != MPI_SUCCESS){
        printf("Error %d\n", err);

        char* str = malloc(STR_DEFAULT_LENGTH * sizeof(char));
        int str_len;

        MPI_Error_string(err, str, &str_len);

        printf("Error: %s, length: %d\n", str, str_len);

        free(str);
    }
}


void get_2_most_closest_multipliers(long int number, int* dividers){
    int sroot = (int) sqrt((double) number);
    for (int cur_div = sroot; cur_div > 0; --cur_div){
        if (number % cur_div == 0){
            dividers[0] = cur_div;
            dividers[1] = number / cur_div;

            return;
        }
    }
    return;
}


void build_matrix_filename(long int n_rows, long int n_cols, char* filename){
    sprintf(filename, "matrix_%ld_%ld.txt", n_rows, n_cols);
    return;
}


void build_vector_filename(long int n_elems, char* filename){
    sprintf(filename, "vector_%ld.txt", n_elems);
    return;
}


void print_matr(double* matr, long int n_rows, long int n_cols, int proc_num){
    printf("%d\tMATRIX (%ld x %ld):\n", proc_num, n_rows, n_cols);
    for (long int i = 0; i < n_rows; ++i){
        for (long int j = 0; j < n_cols; ++j){
            printf("%lf ", matr[i * n_cols + j]);
        }
        printf("\n");
    }
    return;
}


void print_vec(double* vec, long int n_elems, int proc_num){
    printf("%d\tVECTOR:\n", proc_num);
    for (long int i = 0; i < n_elems; ++i){
        printf("%d\t%lf\n", proc_num, vec[i]);
    }
    return;
}


int load_matr(long int n_rows, long int n_cols, double* matrix){
    char* body = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
    build_matrix_filename(n_rows, n_cols, body);
    char filename[MAX_FILENAME_LENGTH] = "./data/";
    strcat(filename, body);
    free(body);
    printf("Reading matrix from file '%s'...\n", filename);

    FILE *fp = fopen(filename, "r");
    if (fp == NULL){
        return -1;
    }

    for (long int i = 0; i < n_rows; ++i){
        for (long int j = 0; j < n_cols; ++j){
            fscanf(fp, "%lf", &matrix[i * n_cols + j]);
        }
    }

    return 0;
}


int load_vec(long int n_rows,  double* vector){
    char* body = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
    build_vector_filename(n_rows, body);
    char filename[MAX_FILENAME_LENGTH] = "./data/";
    strcat(filename, body);
    free(body);
    printf("Reading vector from file '%s'...\n", filename);

    FILE *fp = fopen(filename, "r");
    if (fp == NULL){
        return -1;
    }

    for (long int i = 0; i < n_rows; ++i){
        fscanf(fp, "%lf", &vector[i]);
    }

    return 0;
}


void multiply_std_rowwise(double* matrix, double* vector, long int n_rows, long int n_cols, double* result){
    for (long int i = 0; i < n_rows; ++i){
        double sum = 0;
        for (long int j = 0; j < n_cols; ++j){
            sum += matrix[i * n_cols + j] * vector[j];
        }
        result[i] = sum;
    }

    return;
}
