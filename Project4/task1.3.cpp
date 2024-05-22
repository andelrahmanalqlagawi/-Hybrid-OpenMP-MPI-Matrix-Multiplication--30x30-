// Hybrid openmp+mpi
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
int main(int argc, char** argv)
{
	int rank, size;
	int mtx[30][30], mul[30][30];
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int const row_count = 30 / size;


	if (rank == 0) {

		for (int i = 0; i < 30; i++)
			for (int j = 0; j < 30; j++)
			{
				mtx[i][j] = i + j;
			}
	}
	
		int mat_1[30][6], mat_2[6][30], mul_mat[30][30];
	int arry_start[] = { 0,6,12,18,24 };
	int  arry_size[] = { 6,6,6,6,6 };

	/*int mat_1[30][10], mat_2[10][30], mul_mat[30][30];
	int arry_start[] = { 0,10,20 };
	int  arry_size[] = { 10,10,10 };
*/
	auto start = high_resolution_clock::now();
	//for (int i = 0; i < 30; i++)
	//{
	//	MPI_Scatterv(mtx[i], arry_size, arry_start, MPI_INT, mat_1[i],30*row_count, MPI_INT, 0, MPI_COMM_WORLD);
	//	
	//	
	//}
	MPI_Scatter(mtx, 30 * row_count, MPI_INT, mat_2, 30 * row_count, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(mtx, row_count * 30, MPI_INT, mat_2, row_count * 30, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for
	for (int i = 0; i < 30; i++)
		for (int j = 0; j < 30; j++)
		{
			mul_mat[i][j] = 0;
			for (int k = 0; k < row_count; k++)
				mul_mat[i][j] += mat_1[i][k] * mat_2[k][j];
		}

	MPI_Reduce(mul_mat, mul, 30 * 30, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	auto end = high_resolution_clock::now();
	if (rank == 0) {

		for (int i = 0; i < 30; i++)
		{
			for (int j = 0; j < 30; j++)
			{
				cout << mul[i][j] << "  ";

			}
			cout << endl;
		}
		auto time = duration_cast<microseconds>(end - start);

		cout << "the time taken is: \t" << time.count() << endl;
	}
	MPI_Finalize();
	return 0;
}
