#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// выделяет память под матрицу (строки и столбцы)
int allocMatrix(int*** mat, int rows, int cols) {
	int* base = (int*)malloc(sizeof(int*) * rows * cols);
	if (!base) {
		return -1;
	}
	*mat = (int**)malloc(rows * sizeof(int*));
	if (!mat) {
		free(base);
		return -1;
	}

	for (int i = 0; i < rows; i++) {
		(*mat)[i] = &(base[i * cols]);
	}
	return 0;
}

// освобождает память, выделенную под матрицу
int freeMatrix(int*** mat)
{
	free(&((*mat)[0][0]));
	free(*mat);
	return 0;
}

// умножение матриц
void matrixMultiply(int** x, int** y, int rows, int cols, int*** res) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int sum = 0;
			for (int k = 0; k < rows; k++) {
				sum += x[i][k] * y[k][j];
			}
			(*res)[i][j] = sum;
		}
	}
}

// вывод матрицы в терминал
void printMatrix(int** mat, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%d ", mat[i][j]);
		}
		printf("\n");
	}
}

// запись матрицы в файл
void printMatrixFile(int** mat, int size, FILE* fp) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			fprintf(fp, "%d ", mat[i][j]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char* argv[]) {
	MPI_Comm cart_comm;
	int dims[2], periods[2], allow_reorder;
	int coords[2], proc_id;
	FILE *file_ptr;
	int **matA = NULL, **matB = NULL, **matC = NULL;
	int **blkA = NULL, **blkB = NULL, **blkC = NULL;
	int **tempA = NULL, **tempB = NULL;
	int num_rows = 0;
	int num_cols;
	int total_vals = 0;
	int world_size;
	int grid_dim;
	int block_size;
	int left_nbr, right_nbr, top_nbr, bot_nbr;
	int bcast_buf[4];

	// инициализация MPI
	MPI_Init(NULL, NULL);

	// общее число процессов
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	int my_rank;
	// определение ранга текущего процесса
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0) {
		int val;
		char ch;

		// открытие файла и подсчёт строк и столбцов
		file_ptr = fopen("A.txt", "r");
		if (file_ptr == NULL) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		while (fscanf(file_ptr, "%d", &val) != EOF) {
			ch = fgetc(file_ptr);
			if (ch == '\n') {
				num_rows = num_rows + 1;
			}
			total_vals++;
		}
		num_cols = total_vals / num_rows;
		printf("строк = %d, столбцов = %d\n", num_rows, num_cols);

		//матрица должна быть квадратной
		if (num_cols != num_rows) {
			printf("Матрица должна быть квадратной!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		double root_p = sqrt(world_size);
		//число процессов должно быть полным квадратом
		if ((root_p - floor(root_p)) != 0) {
			printf("Число процессов должно быть полным квадратом!\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		int int_root = (int)root_p;
		// размерность должна делиться на число процессов
		if (num_cols % int_root != 0 || num_rows % int_root != 0) {
			printf("Размерность не делится на %d!\n", int_root);
			MPI_Abort(MPI_COMM_WORLD, 3);
		}
		grid_dim = int_root;
		block_size = num_cols / int_root;

		fseek(file_ptr, 0, SEEK_SET);
		// выделение памяти под матрицы для умножения
		if (allocMatrix(&matA, num_rows, num_cols) != 0) {
			printf("Не удалось выделить память для матрицы A!\n");
			MPI_Abort(MPI_COMM_WORLD, 4);
		}
		if (allocMatrix(&matB, num_rows, num_cols) != 0) {
			printf("Не удалось выделить память для матрицы B!\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}

		// чтение первой матрицы из файла
		for (int i = 0; i < num_rows; i++) {
			for (int j = 0; j < num_cols; j++) {
				fscanf(file_ptr, "%d", &val);
				matA[i][j] = val;
			}
		}
		printf("Матрица A:\n");
		printMatrix(matA, num_rows);
		fclose(file_ptr);

		// чтение второй матрицы из файла
		file_ptr = fopen("B.txt", "r");
		if (file_ptr == NULL) {
			return 1;
		}
		for (int i = 0; i < num_rows; i++) {
			for (int j = 0; j < num_cols; j++) {
				fscanf(file_ptr, "%d", &val);
				matB[i][j] = val;
			}
		}
		printf("Матрица B:\n");
		printMatrix(matB, num_rows);
		fclose(file_ptr);
		// выделение памяти под результирующую матрицу
		if (allocMatrix(&matC, num_rows, num_cols) != 0) {
			printf("Не удалось выделить память для матрицы C!\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}

		bcast_buf[0] = grid_dim;
		bcast_buf[1] = block_size;
		bcast_buf[2] = num_rows;
		bcast_buf[3] = num_cols;
	}
	

	// создание 2D декартовой решётки процессов
	MPI_Bcast(&bcast_buf, 4, MPI_INT, 0, MPI_COMM_WORLD);
	grid_dim = bcast_buf[0];
	block_size = bcast_buf[1];
	num_rows = bcast_buf[2];
	num_cols = bcast_buf[3];

	dims[0] = grid_dim; dims[1] = grid_dim;
	periods[0] = 1; periods[1] = 1;
	allow_reorder = 1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, allow_reorder, &cart_comm);

	// выделение памяти под локальные блоки матриц A и B
	allocMatrix(&blkA, block_size, block_size);
	allocMatrix(&blkB, block_size, block_size);

	// создание пользовательского типа данных для подматриц
	int global_shape[2] = { num_rows, num_cols };
	int local_shape[2] = { block_size, block_size };
	int offsets[2] = { 0, 0 };
	MPI_Datatype base_type, resized_type;
	MPI_Type_create_subarray(2, global_shape, local_shape, offsets, MPI_ORDER_C, MPI_INT, &base_type);
	MPI_Type_create_resized(base_type, 0, block_size * sizeof(int), &resized_type);
	MPI_Type_commit(&resized_type);

	int *ptrA = NULL;
	int *ptrB = NULL;
	int *ptrC = NULL;
	if (my_rank == 0) {
		ptrA = &(matA[0][0]);
		ptrB = &(matB[0][0]);
		ptrC = &(matC[0][0]);
	}

	// разброс массива всем процессам
	int* send_counts = (int*)malloc(sizeof(int) * world_size);
	int* displs = (int*)malloc(sizeof(int) * world_size);

	if (my_rank == 0) {
		for (int i = 0; i < world_size; i++) {
			send_counts[i] = 1;
		}
		int disp = 0;
		for (int i = 0; i < grid_dim; i++) {
			for (int j = 0; j < grid_dim; j++) {
				displs[i * grid_dim + j] = disp;
				disp += 1;
			}
			disp += (block_size - 1) * grid_dim;
		}
	}

	MPI_Scatterv(ptrA, send_counts, displs, resized_type, &(blkA[0][0]),
		num_rows * num_cols / (world_size), MPI_INT,
		0, MPI_COMM_WORLD);
	MPI_Scatterv(ptrB, send_counts, displs, resized_type, &(blkB[0][0]),
		num_rows * num_cols / (world_size), MPI_INT,
		0, MPI_COMM_WORLD);

	if (allocMatrix(&blkC, block_size, block_size) != 0) {
		printf("Не удалось выделить память для локальной матрицы C на процессе %d!\n", my_rank);
		MPI_Abort(MPI_COMM_WORLD, 7);
	}

	// начальный сдвиг (pre-skew)
	MPI_Cart_coords(cart_comm, my_rank, 2, coords);
	MPI_Cart_shift(cart_comm, 1, coords[0], &left_nbr, &right_nbr);
	MPI_Sendrecv_replace(&(blkA[0][0]), block_size * block_size, MPI_INT, left_nbr, 1, right_nbr, 1, cart_comm, MPI_STATUS_IGNORE);
	MPI_Cart_shift(cart_comm, 0, coords[1], &top_nbr, &bot_nbr);
	MPI_Sendrecv_replace(&(blkB[0][0]), block_size * block_size, MPI_INT, top_nbr, 1, bot_nbr, 1, cart_comm, MPI_STATUS_IGNORE);

	// инициализация результирующей матрицы нулями
	for (int i = 0; i < block_size; i++) {
		for (int j = 0; j < block_size; j++) {
			blkC[i][j] = 0;
		}
	}

	int** tmp_res = NULL;
	if (allocMatrix(&tmp_res, block_size, block_size) != 0) {
		printf("Не удалось выделить память для временной матрицы на процессе %d!\n", my_rank);
		MPI_Abort(MPI_COMM_WORLD, 8);
	}

	// умножение матриц по алгоритму Кэннона
	double start_time = MPI_Wtime();
	for (int step = 0; step < grid_dim; step++) {
		matrixMultiply(blkA, blkB, block_size, block_size, &tmp_res);

		for (int i = 0; i < block_size; i++) {
			for (int j = 0; j < block_size; j++) {
				blkC[i][j] += tmp_res[i][j];
			}
		}
		// сдвиг A влево и B вверх
		MPI_Cart_shift(cart_comm, 1, 1, &left_nbr, &right_nbr);
		MPI_Cart_shift(cart_comm, 0, 1, &top_nbr, &bot_nbr);
		MPI_Sendrecv_replace(&(blkA[0][0]), block_size * block_size, MPI_INT, left_nbr, 1, right_nbr, 1, cart_comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(&(blkB[0][0]), block_size * block_size, MPI_INT, top_nbr, 1, bot_nbr, 1, cart_comm, MPI_STATUS_IGNORE);
	}
	
	// собираем результаты на нулевом процессе
	MPI_Gatherv(&(blkC[0][0]), num_rows * num_cols / world_size, MPI_INT,
		ptrC, send_counts, displs, resized_type,
		0, MPI_COMM_WORLD);

	double end_time = MPI_Wtime();

	// освобождение вспомогательных матриц
	freeMatrix(&blkC);
	freeMatrix(&tmp_res);

	// открытие файла для записи результата
	FILE * out_file = fopen("output.txt", "w");

	// вывод результата
	if (my_rank == 0) {
		printf("Результат C:\n");
		printMatrix(matC, num_rows);
		printf("\nВремя выполнения: %lf\n", end_time - start_time);
		printMatrixFile(matC, num_rows, out_file);
		fprintf(out_file, "\nВремя выполнения: %lf\n", end_time - start_time);
	}
	
	// завершение работы MPI
	MPI_Finalize();

	return 0;
}
