#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

// Constants
#define MAX_FILENAME_LENGTH 128
#define MAIN_PROCESS 0
#define STR_DEFAULT_LENGTH 128
#define SUBMATR_TAG 15
#define SUBVEC_TAG 25
#define N_DIVIDERS 2

// Utils
void process_error(int err);
void get_2_most_closest_multipliers(long int number, int* dividers);

// Matrix utils
void build_matrix_filename(long int n_rows, long int n_cols, char* filename);
void build_vector_filename(long int n_elems, char* filename);
void print_matr(double* matr, long int n_rows, long int n_cols, int proc_num);
void print_vec(double* vec, long int n_elems, int proc_num);
int load_matr(long int n_rows, long int n_cols, double* matrix);
int load_vec(long int n_rows,  double* vector);
void multiply_std_rowwise(double* matrix, double* vector, long int n_rows, long int n_cols, double* result);

#endif // UTILS_H_INCLUDED
