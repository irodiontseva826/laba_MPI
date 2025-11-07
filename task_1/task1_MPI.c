 
//Компиляция:
 //mpicc -O2 -std=c11 -o task1_mpi_pi task1_MPI.c -lm          
 //Запуск:
 //mpirun -np 4 ./task1_mpi_pi 1000 result.csv             where  4 - number of processes, 1000 - number of trials
 
 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <math.h>
 
 #ifndef M_PI
 #   define M_PI 3.14159265358979323846
 #endif
 
 static inline double rand01(void)
 {
     return (double)rand() / (double)RAND_MAX;
 }


 
 int main(int argc, char *argv[])
 {
     int rank, size;
     MPI_Init(&argc, &argv); 
     MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     long long total_trials = atoll(argv[1]); 
     if (total_trials <= 0) {
         if (rank == 0)
             fprintf(stderr, "Number of trials must be positive.\\n");
         MPI_Finalize();
         return EXIT_FAILURE;
     }

 
     const char *out_fname = "result.csv";
     if (argc == 3)
        out_fname = argv[2]; 

     long long trials_per_proc = total_trials / size;  
     long long remainder = total_trials % size; 
     long long local_trials = trials_per_proc + (rank < remainder ? 1 : 0); 

     unsigned int seed = (unsigned int)time(NULL) + (unsigned int)rank * 137;  
     srand(seed);
 
     MPI_Barrier(MPI_COMM_WORLD); 
     double t_start = MPI_Wtime(); 
 
     long long local_inside = 0; 
     for (long long i = 0; i < local_trials; ++i) {
         double x = 2.0 * rand01() - 1.0;   
         double y = 2.0 * rand01() - 1.0;  
         if (x * x + y * y <= 1.0)
             ++local_inside;
     }
 
     long long global_inside = 0;
     MPI_Reduce(&local_inside, &global_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 
 
     MPI_Barrier(MPI_COMM_WORLD); 
     double t_end = MPI_Wtime(); 
     double local_elapsed = t_end - t_start;
 
     double max_elapsed = 0.0;
     MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

     if (rank == 0) {
         double ratio   = (double)global_inside / (double)total_trials; 
         double pi_est  = 4.0 * ratio; 
         double abs_err = fabs(M_PI - pi_est); 

        int file_exists = 0;
        FILE *check = fopen(out_fname, "r");
        if (check != NULL) {
            file_exists = 1;
            fclose(check);
        }

        FILE *f = fopen(out_fname, "a");
        if (f == NULL) {
            fprintf(stderr, "Cannot open output file \"%s\"\n", out_fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (!file_exists) {
            fprintf(f, "MPI_processes,Total_trials,Points_inside,Ratio,Estimated_pi,Absolute_error,Execution_time\n");
        }

        fprintf(f, "%d,%lld,%lld,%.12f,%.12f,%.12f,%.6f\n",
            size, total_trials, global_inside,
            ratio, pi_est, abs_err, max_elapsed);

        fclose(f);
        printf("Result appended to \"%s\"\n", out_fname);

    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
