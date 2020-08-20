#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
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

// calculates the determinant of input matrix, be finding U matrix in LU factorization
double determinant(double *matrix, int n) {

	// a variable to track the number of times we've swapped rows
	int swap = 1;

	// determinant
	double det = 1.0;

	// looping through diagonal entries
	for (int i = 0; i < n; i++) {

		// finding a non-zero entry below current diagonal entry
		int k = i;
		while (matrix[k * n + i] == 0 && k < n)
			k++;

		// if nothing was found, then determinant is 0
		if (matrix[k * n + i] == 0)
			return 0;

		// if k != i, then swap the rows
		if (k != i) {
			swap *= -1;
			for (int j = 0; j < n; j++) {
				double temp = matrix[i * n + j];
				matrix[i * n + j] = matrix[k * n +j];
				matrix[k * n + j] = temp;
			}
		}

		det *= matrix[i * n + i];

		// making the entries below diagonal entry 0, using row operations
		for (int j = i + 1; j < n; j++) {

			double factor = -1 * matrix[j * n + i] / matrix[i * n + i];
			matrix[j * n + i] = 0;

			for (int k = i + 1; k < n; k++) {
				matrix[j * n + k] += factor * matrix[i * n + k];
			}
		}
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

	// closing the files;
	fclose(ifp);
	fclose(ofp);

}

int main() {

	double startTime = omp_get_wtime();

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

				// process the file
				processFile(fileName);
			}
		}
		closedir(d);
	}

	printf("%lfs\n", omp_get_wtime() - startTime);
}
