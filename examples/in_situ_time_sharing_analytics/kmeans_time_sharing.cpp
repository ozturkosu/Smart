#include <chrono>
#include <memory>
#include <mpi.h>
#include <iostream>

#include "../common_app_headers/kmeans.h"
#include "../../include/scheduler.h"

#define NUM_THREADS 4  // The # of threads for analytics task.
// For k-means application, STEP and NUM_DIMS in kmeans.h must be equal. 
#define STEP 8
#define NUM_DIMS  8// The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 256 // The total number of elements of the simulated data.
#define NUM_ITERS 2  // The # of iterations.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT 1

using namespace std;

//Some functions for Jacobi -- Emin 

void load_data(double** A, double* F, double* X, int DIMENSION)
{
    srand(time(0));

    for (int i = 0; i < DIMENSION; i++) {
        A[i] = new double[DIMENSION];

        for (int j = 0; j < DIMENSION; j++) {
            if (i == j)
                A[i][j] = 100 * DIMENSION + rand() % 300 * DIMENSION;
            else
                A[i][j] = 1 + rand() % 10;
        }
        F[i] = 1 + rand() % 10;
        X[i] = 1;
    }
    cout << "Dataload finished!" << endl;
}

/// N - размерность матрицы; A[N][N] - матрица коэффициентов, F[N] - столбец свободных членов,
/// X[N] - начальное приближение, также ответ записывается в X[N];

void solve_worker(double** chunkA, double* chunkX, double* chunkF, int iternum, int N, int chunklen, int rank_new , double* in )
{
    cout << "Worker launched from process #" << rank_new << endl;
    cout << chunkA[0][0] << endl;
    cout << chunkF[0] << endl;
    cout << chunkX[0] << endl;
    double* TempX = new double[chunklen];

    cout << "Worker launched from process #" << rank_new << endl;
    for (int run = 0; run < iternum; run++)
    {
        for (int i = 0; i < chunklen; i++)
        {
            TempX[i] = chunkF[i];
            for (int g = 0; g < N; g++)
            {
                if (i != g)
                    TempX[i] -= chunkA[i][g] * chunkX[g];
            }
            TempX[i] /= chunkA[i][i];
        }
    }
    cout << "Worker #" << rank_new << " has completed normally" << endl;

    for (size_t i = 0; i < chunklen; i++)
    {
        in[i + (chunklen* rank_new)] = TempX[i] ;
    }
}



// *********************************


int main(int argc, char* argv[]) {

  // MPI initialization.
  int mpi_status = MPI_Init(&argc, &argv);
  if (mpi_status != MPI_SUCCESS) {
    printf("Failed to initialize MPI environment.\n");
    MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Only used for time statistics, not necessarily added to the simulation code.
  chrono::time_point<chrono::system_clock> clk_beg, clk_end;
  clk_beg = chrono::system_clock::now();

  const size_t total_len = NUM_ELEMS;
  double* in = new double[total_len];

  // The output is a 2D array that indicates k vectors in a multi-dimensional
  // space.
  const size_t out_len = NUM_MEANS;
  double** out = new double*[out_len];
  for (size_t i = 0; i < out_len; ++i) {
    out[i] = new double[NUM_DIMS];
  }



  //**********************************

  /*
  // Run the given simulation.
  for (size_t i = 0; i < total_len; ++i) {
    in[i] = i + rank;
  }
  */

  int size ;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank_new;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_new);

  int iterations = 5;       ////////////////////////
  int dimension = 256;     ///////////////////////
  int startIndex = 0;
  int endIndex = 0;
  int chunk = dimension / (size - 1);


  if (dimension % (size - 1) != 0) {
      MPI_Finalize();
      cout << "Process number must be divisor of dimension!";
      return 0;
  }


  if (rank_new == 0)
  {
            double** A = new double*[dimension];//Matrix2D A(dimension);  //N*N
            double* F = new double[dimension];//Matrix1D F(dimension);  //N
            double* X = new double[dimension];//Matrix1D X(dimension);  //N

            cout << "Main, total processors: " << size << endl;
            load_data(A, F, X, dimension);

            for (int i = 1; i < size; i++)
            {   // Set task for workers
                double** chunkA = new double*[chunk];

                for (int b = 0; b < chunk; b++) {
                    chunkA[b] = new double[dimension]; //.resize(dimension);
                }
                double* chunkF = new double[dimension];
                double* chunkX = new double[dimension];

                startIndex = chunk * (i - 1);
                endIndex = chunk * i - 1;

                for (int j = startIndex; j <= endIndex; j++)
                {
                    for (int x = 0; x < dimension; x++)
                    {
                        chunkA[j - startIndex][x] = A[j][x];
                    }
                    chunkF[j - startIndex] = F[j];
                    chunkX[j - startIndex] = X[j];
                }
                cout << "Master sends to process #" << i << " array[start, end]: " << startIndex << " - " << endIndex << endl;
                MPI_Send(&(chunkA[0][0]), chunk*dimension, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                cout << "Master sent to process # array A" << endl;
                MPI_Send(&(chunkF[0]), chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                cout << "Master sent to process # array F" << endl;
                MPI_Send(&(chunkX[0]), chunk, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                cout << "Master sent to process # array X" << endl;
                cout << "Send for #" << i << " is OK!" << endl;

            }
  }
  else if (rank_new != 0)
  {
        cout << "Process " << rank_new << "reached this code" << endl;
        double** chunkA = new double*[chunk];
        for (int b = 0; b < chunk; b++) {
                chunkA[b] = new double[dimension];
        }
        double* chunkF = new double[dimension];
        double* chunkX = new double[dimension];

        cout << "Process #"<< rank_new <<" -- Declaring buffers is ok!" << endl;

        MPI_Recv(&(chunkA[0][0]), chunk*dimension, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(chunkF[0]), chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(chunkX[0]), chunk, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Process #" << rank_new << " received values. " << endl;

        solve_worker(chunkA, chunkX, chunkF, iterations, dimension, rank_new, chunk, in);\
        cout << "Process #" << rank_new << " completed. " << endl;
 }

 MPI_Barrier(MPI_COMM_WORLD); 

 printf("Jacobi is done.\n");
 
 


  //**************************************

  // Set up the initial k centroids.
  double** means = new double*[NUM_MEANS];
  for (int i = 0; i < NUM_MEANS; ++i) {
    means[i] = new double[NUM_DIMS];
  }
  for (int i = 0; i < NUM_MEANS; ++i) {
    for (int j = 0; j < NUM_DIMS; ++j) {
      means[i][j] = i * 10;  // This setting can result in some empty clusters.
    }
  }

  clk_end = chrono::system_clock::now();
  chrono::duration<double> sim_seconds = clk_end - clk_beg;
  if (rank == 0) {
    printf("Simulation time = %.2f secs.\n", sim_seconds.count());
    printf("Simulation data is ready...\n");
  }

  // Insert in-situ processing code.
  if (rank == 0)
    printf("Run in-situ processing...\n");
  SchedArgs args(NUM_THREADS, STEP, (void*)means, NUM_ITERS);
  unique_ptr<Scheduler<double, double*>> kmeans(new Kmeans<double>(args));   
  kmeans->set_red_obj_size(sizeof(ClusterObj<double>));
  kmeans->run(in, total_len, out, out_len);
  if (rank == 0)
    printf("In-situ processing is done.\n");

  // Print out the combination map if required.
  if (PRINT_COMBINATION_MAP && rank == 0) {
    printf("\n");
    kmeans->dump_combination_map();
  }

  // Print out the final result on the master node if required.
  if (PRINT_OUTPUT && rank == 0) {
    printf("Final output on the master node:\n");
    for (size_t i = 0; i < out_len; ++i) {
      for (int j = 0; j < NUM_DIMS; ++j) {
        printf("%.2f ", out[i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }

  // Only used for time statistics, not necessarily added to the simulation code.
  clk_end = chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds = clk_end - clk_beg;
  if (rank == 0)
    printf("Analytics time = %.2f secs.\n", elapsed_seconds.count() - sim_seconds.count());
  printf("Total processing time on node %d = %.2f secs.\n", rank, elapsed_seconds.count());

  delete [] in;
  for (size_t i = 0; i < out_len; ++i) {
    delete [] out[i];
  }
  delete [] out;

  for (int i = 0; i < NUM_MEANS; ++i) {
    delete [] means[i];
  }
  delete [] means;

  MPI_Finalize();

  return 0;
}
