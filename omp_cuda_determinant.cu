#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>


#define LINE_BUFFER_SIZE 200000


// prints the given square matrix
void print_matrix(double *matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%lf ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

int DecomposeLU(double *matrix, int n) {

	double epsilon = 0.000000001;
	int swap = 1;

	cublasStatus_t cublasStatus;

	// creating cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// creating CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// setting the stream
	cublasSetStream(handle, stream);


	for (int k = 0; k < n - 1; k++) {

		// finding the pivot row
		int pivotRow;
		cublasStatus = cublasIdamax(handle, n - k, matrix + k + k * n, 1, &pivotRow);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
			printf ("cublasIdamax failed!");
			exit(-1);
		}

		pivotRow += k - 1;
		int kp1 = k + 1;

		// getting the pivot row to the top
		if (pivotRow != k) {
			swap *= -1;
			cublasStatus = cublasDswap(handle, n, matrix + pivotRow, n, matrix + k, n);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
				printf ("cublasDswap failed!");
				exit(-1);
			}
		}

		// checking if we got 0 on the diagonal entry
		double valcheck;
		cublasStatus = cublasGetVector(1, sizeof(double), matrix + k + k * n, 1, &valcheck, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
			printf ("cublasGetVector failed!");
			exit(-1);
		}


		if (fabs(valcheck) < epsilon)
		   return swap;

		// finding partial L and U
		if (kp1 < n) {
			const double alpha = 1.0f / valcheck;
			cublasStatus = cublasDscal(handle, n - kp1, &alpha , matrix + kp1 + k * n, 1);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
				printf ("cublasDscal failed!");
				exit(-1);
			}
		}

		if (kp1 < n) {
			const double alpha = -1.0f;
			cublasStatus = cublasDger(handle, n - kp1, n - kp1, &alpha, matrix + kp1 + k * n, 1, matrix + k + kp1 * n, n, matrix + kp1 * n + kp1, n);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
				printf ("cublasDger failed!");
				exit(-1);
			}
		}

	}

	// destroying the stream
	cudaStreamDestroy(stream);

	return swap;

}

// calculates the determinant of input matrix, using Row-Reduction algorithm
double determinant(double *matrix, int n) {

	// allocating memory for device matrix
	double *d_matrix;

	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**) &d_matrix, n * n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		printf("allocating device memory for the matrix failed!");
		exit(-1);
	}

	// copying the h_matrix into d_matrix
	cudaStatus = cudaMemcpy(d_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("copying matrix into d_matrix failed!");
		exit(-1);
	}

	// LU decomposition
	int swap = DecomposeLU(d_matrix, n);

	cudaStatus = cudaMemcpy(matrix, d_matrix, n * n * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("getting matrix from device failed!");
		exit(-1);
	}

	double det = 1.0;
	for (int i = 0; i < n; i++) {
		det *= matrix[i * n + i];
	}

	// taking the number of swaps into consideration
	det *= swap;

	return det;
}


void processFile(char *fileName) {

	// making input file path
	char inputFilePath[40] = "data_in/";
	strcat(inputFilePath, fileName);

	// opening the input file
	FILE *ifp;
	ifp = fopen(inputFilePath, "r");

	// making input file path
	char outputFilePath[40] = "data_out/";
	strcat(outputFilePath, fileName);

	// creating the empty output file
	FILE *ofp;
	ofp = fopen(outputFilePath, "w");

	// line buffer
	char line[LINE_BUFFER_SIZE];

	// number of matrices read so far
	int m = 0;

	while (1) {

		// reading '\n' between the matrices
		if (m > 0) {
			char *ret = fgets(line, LINE_BUFFER_SIZE, ifp);
			// checking if we've reached the end of the file
			if (ret == NULL)
				break;
		}

		// reading the first line of the current matrix
		char *ret = fgets(line, LINE_BUFFER_SIZE, ifp);

		// checking if we've reached the end of the file
		if (ret == NULL)
			break;

		// finding the size of the current matrix by counting the ' ' characters
		int n = 0;
		for (int i = 0; i < LINE_BUFFER_SIZE; i++) {
			if (line[i] == '\n')
				break;
			if (line[i] == ' ')
				n++;
		}

		// allocating memory for the current matrix
		double *matrix = (double*) malloc(n * n * sizeof(double));

		// storing the first line elements into the matrix
		char *savePtr;
		char *token = strtok_r(line, " ", &savePtr);
		for (int i = 0; i < n; i++) {
			double d;
			sscanf(token, "%lf", &d);
			matrix[i] = d;
			token = strtok_r(NULL, " ", &savePtr);
		}

		// reading the rest of the file
		for (int i = 1; i < n; i++) {
			fgets(line, LINE_BUFFER_SIZE, ifp);

			char * token = strtok_r(line, " ", &savePtr);
			for (int j = 0; j < n; j++) {
				double d = 1.0;
				sscanf(token, "%lf", &d);
				matrix[i * n + j] = d;
				token = strtok_r(NULL, " ", &savePtr);
			}
		}

		// calculating the determinant
		double det = determinant(matrix, n);

		// writing the result in output file
		fprintf(ofp, "%lf\n", det);

		free(matrix);
		m++;

	}

	// closing the files
	fclose(ifp);
	fclose(ofp);

}

int main() {

	double startTime = omp_get_wtime();

	#pragma omp parallel
	{
		#pragma omp single nowait
		{
			// listing all the input files in 'data_in' folder
			DIR *d;
			struct dirent *dir;
			d = opendir("data_in");

			if (d) {

				// for every file in 'data_in'
				while ((dir = readdir(d)) != NULL) {
					if (strcmp(dir->d_name, ".") != 0 && strcmp(dir->d_name, "..") != 0) {

						char fileName[30] = "";
						strcat(fileName, dir->d_name);

						// create a task for the file to be done by a thread
						#pragma omp task firstprivate(fileName)
						processFile(fileName);
					}
				}
				closedir(d);
			}
		}
	}

	printf("%fs\n", omp_get_wtime() - startTime);
}
