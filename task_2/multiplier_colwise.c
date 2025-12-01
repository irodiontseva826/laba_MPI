#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"


void distribute_data (double* matrix, double* vector, long int n_rows, long int n_cols, long int local_n, int my_rank, int comm_sz, MPI_Comm comm, double* local_matr, double* local_vec){

    MPI_Datatype mpi_vec;

    MPI_Type_vector(
        n_rows,
        local_n,
        n_cols,
        MPI_DOUBLE,
        &mpi_vec
    );

    MPI_Type_commit(&mpi_vec);

    int error;

    if (my_rank == MAIN_PROCESS){
        int position;

        int pack_size;
        MPI_Pack_size(1, mpi_vec, MPI_COMM_WORLD, &pack_size);

        for (int i = 1; i < comm_sz; ++i){

            position = 0;

            error = MPI_Pack(
                &matrix[i * local_n],
                1,
                mpi_vec,
                local_matr,
                pack_size,
                &position,
                MPI_COMM_WORLD
            );
            process_error(error);

            error = MPI_Send(
                local_matr,
                n_rows * local_n,
                MPI_DOUBLE,
                i,
                SUBMATR_TAG,
                MPI_COMM_WORLD
            );
            process_error(error);
        }

        position = 0;

        error = MPI_Pack(
            &matrix[0],
            1,
            mpi_vec,
            local_matr,
            pack_size,
            &position,
            MPI_COMM_WORLD
        );
        process_error(error);

    }
    else {
        error = MPI_Recv(
            &local_matr[0],
            n_rows * local_n,
            MPI_DOUBLE,
            MAIN_PROCESS,
            SUBMATR_TAG,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        process_error(error);
    }

    error = MPI_Scatter(
        vector,
        local_n,
        MPI_DOUBLE,
        local_vec,
        local_n,
        MPI_DOUBLE,
        MAIN_PROCESS,
        comm
    );
    process_error(error);

    error = MPI_Type_free(&mpi_vec);
    process_error(error);

    return;
}


void multiply_colwise(double* local_matr, double* local_vec, long int n_rows, long int local_n, int my_rank, int comm_sz, double* result){

    for (long int j = 0; j < local_n; ++j){
        for (long int i = 0; i < n_rows; ++i){
            local_matr[i * local_n + j] *= local_vec[j];
        }
    }

    long int n_cols = local_n * comm_sz;

    double* columns = (double*) malloc(n_cols * sizeof(double));
    for (long int i = 0; i < n_rows; ++i){
        double sum = 0.0;
        for (long int j = 0; j < local_n; ++j){
            sum += local_matr[i * local_n + j];
        }
        columns[i] = sum;
    }

    MPI_Reduce(columns, result, n_rows, MPI_DOUBLE, MPI_SUM, MAIN_PROCESS, MPI_COMM_WORLD);

    free(columns);

    return;
}



int main(int argc, char** argv){
    int error;
    double sum_time;

    long int n_rows = strtol(argv[1], NULL, 10);
    long int n_cols = strtol(argv[2], NULL, 10);

    int comm_sz;
    int my_rank;

    double start, finish, elapsed = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == MAIN_PROCESS){
        if (n_cols % comm_sz != 0){
            printf("\nERROR!!!\n%ld mod %d = %ld. Unable to parallellize task.\n", n_cols, comm_sz, n_rows % comm_sz);
            return 0;
        }

        char* new_file_name = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
        sprintf(new_file_name, "./data/out/colwise.csv");

        if (fopen(new_file_name, "r") == NULL){
            FILE* fp = fopen(new_file_name, "w");
            if (fp == NULL){
                printf("Unable to create output file.\n");
                return 0;
            }
            fprintf(fp, "n_rows, n_cols, n_processes, time\n");
            fclose(fp);
        }

        sum_time = 0;
    }

    long int local_n = n_cols / comm_sz;


    double* matrix;
    double* vector;
    double* result;


    if (my_rank == MAIN_PROCESS){
        printf("n_rows = %ld\n", n_rows);
        printf("n_cols = %ld\n", n_cols);
        printf("local_n = %ld\n", local_n);
        printf("comm_sz = %d\n", comm_sz);
        printf("my_rank = %d\n", my_rank);

        matrix = (double*) malloc(n_rows * n_cols * sizeof(double));
        vector = (double*) malloc(n_cols * sizeof(double));
        result = (double*) malloc(n_rows * sizeof(double));


        if (my_rank == MAIN_PROCESS){
            error = load_matr(n_rows, n_cols, matrix);
            if (error == -1){
                char* filename = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
                build_matrix_filename(n_rows, n_cols, filename);
                printf("Unable to locate matrix file '%s'\n", filename);
                free(filename);
                return 0;
            }
        }

        if (my_rank == MAIN_PROCESS){
            error = load_vec(n_cols, vector);
            if (error == -1){
                char* filename = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
                build_vector_filename(n_cols, filename);
                printf("Unable to locate vector file '%s'\n", filename);
                free(filename);
                return 0;
            }
        }
    }

    double* local_matr = malloc(local_n * n_rows * sizeof(double));
    double* local_vec = malloc(local_n * sizeof(double));

    for (int i = 0; i < 100; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        distribute_data(matrix, vector, n_rows, n_cols, local_n, my_rank, comm_sz, MPI_COMM_WORLD, local_matr, local_vec);
        multiply_colwise(local_matr, local_vec, n_rows, local_n, my_rank, comm_sz, result);

        MPI_Barrier(MPI_COMM_WORLD);
        finish = MPI_Wtime();

        double local_elapsed = finish - start;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MAIN_PROCESS, MPI_COMM_WORLD);

        if (my_rank == MAIN_PROCESS)
            sum_time += elapsed;
    }

    free(local_matr);
    free(local_vec);

    MPI_Finalize();

    if (my_rank == MAIN_PROCESS){

        char* new_filename = (char*) malloc(MAX_FILENAME_LENGTH * sizeof(char));
        sprintf(new_filename, "./data/out/colwise.csv");

        FILE* fp = fopen(new_filename, "a+");
        if (fp == NULL){
            printf("Unable to open output file.\n");
            return 0;
        }
        fprintf(fp, "%ld, %ld, %d, %lf\n", n_rows, n_cols, comm_sz, sum_time / 100);

        fclose(fp);

        free(result);
        free(vector);
        free(matrix);
    }

    return 0;
}
