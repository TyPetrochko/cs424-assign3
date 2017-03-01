#include "timing.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include "mpi.h"

#define NUM_RUNS 5
#define MIN(a,b) (((a)<(b))?(a):(b))

double buffer_init(int blockwidth, int N, double *Ablock, double *Bblock, double *Cblock){
  Ablock = (double *) calloc(blockwidth * N, sizeof(double)); /* there is some extra space here */
  Bblock = (double *) calloc(blockwidth * N, sizeof(double));
  Cblock = (double *) calloc(blockwidth * N, sizeof(double));
}

double matrix_init(int, double*, double*, double*);

double master(int, double*, double*, double*); /* do master work and report time */

void worker(int, double*, double*, double*); /* do worker work */

int main(int argc, char **argv) {

  /*
    This is the serial main program for CPSC424/524 Assignment #3.

    Author: Andrew Sherman, Yale University

    Date: 2/01/2016

  */

  int N, i, j, k, run;
  double *A, *B, *C;
  double *Ablock, *Bblock, *Cblock; /* worker buffers */
  int sizeAB, sizeC, iA, iB, iC;

  // make sure to change NUM_RUNS along with this!
  int sizes[NUM_RUNS]={1000,2000,4000,8000,12000};

  double wctime;
  
  MPI_Status status;

  MPI_Init(&argc,&argv); // Required MPI initialization call

  MPI_Comm_size(MPI_COMM_WORLD,&size); // Get no. of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Which process am I?
  
  if (rank == 0) { /* master */
    printf("Matrix multiplication times:\n   N      TIME (secs)\n -----   -------------\n");

    // do all master runs
    for (run=0; run<NUM_RUNS; run++) {
      N = sizes[run];
      matrix_init(N, A, B, C);
      buffer_init(N / size, N, Ablock, Bblock, Cblock);
      MPI_Barrier(MPI_COMM_WORLD);
      master(N, A, B, C);
      
      // cleanup
      free(A); free(Ablock);
      free(B); free(Bblock);
      free(C); free(Cblock);
    }
   
  } else { /* worker */
    for (run=0; run<NUM_RUNS; run++) {
      // init buffers
      N = sizes[run];
      buffer_init(N / size, N, Ablock, Bblock, Cblock);
      
      // wait to start
      MPI_Barrier(MPI_COMM_WORLD);
      worker(N, A, B, C);

      // cleanup
      free(Ablock); 
      free(Bblock); 
      free(Cblock); 
    }
  }
  
  MPI_Finalize(); // Required MPI termination call
}

double matmul_blocking(int N, double* A, double* B, double* C) {

/*
  This is the serial version of triangular matrix multiplication for CPSC424/524 Assignment #3.

  Author: Andrew Sherman, Yale University

  Date: 2/01/2016

*/

  int i, j, k;
  int iA, iB, iC;
  double wctime0, wctime1, cputime;

  timing(&wctime0, &cputime);

  MPI_Status status;

  MPI_Init(&argc,&argv); // Required MPI initialization call

  MPI_Comm_size(MPI_COMM_WORLD,&size); // Get no. of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Which process am I?

  // calculate p
  // initialize a block-row buffer (A), block-column buffer (B),
  //  block-row buffer (C)
  
  /* If I am the master (rank 0) ... */
  if (rank == 0) {
    // do all the stuff from serial main!
    // i.e. generate the random matrix itself
    // barrier
    // start timing
    // 
  }
  else {
    // clear (RANK)th C block-row buffer
    // barrier
    // wait for (RANK)th A block-row assignment
    // 
  }


// This loop computes the matrix-matrix product
//  iC = 0;
//  for (i=0; i<N; i++) {
//    iA = i*(i+1)/2; // initializes row pointer in A
//    for (j=0; j<N; j++,iC++) {
//      iB = j*(j+1)/2; // initializes column pointer in B
//      C[iC] = 0.;
//      for (k=0; k<=MIN(i,j); k++) C[iC] += A[iA+k] * B[iB+k]; // avoids using known-0 entries 
//    }
//  }

  timing(&wctime1, &cputime);
  MPI_Finalize(); // Required MPI termination call
  return(wctime1 - wctime0);
}

void matrix_init(int N, double *A, double *B, double *C){
  int sizeAB, sizeC, i;

  sizeAB = N*(N+1)/2; //Only enough space for the nonzero portions of the matrices
  sizeC = N*N; // All of C will be nonzero, in general!

  A = (double *) calloc(sizeAB, sizeof(double));
  B = (double *) calloc(sizeAB, sizeof(double));
  C = (double *) calloc(sizeC, sizeof(double));

  srand(12345); // Use a standard seed value for reproducibility

  // This assumes A is stored by rows, and B is stored by columns. Other storage schemes are permitted
  for (i=0; i<sizeAB; i++) A[i] = ((double) rand()/(double)RAND_MAX);
  for (i=0; i<sizeAB; i++) B[i] = ((double) rand()/(double)RAND_MAX);
  
  MPI_Barrier(MPI_COMM_WORLD);

  wctime = matmul(N, A, B, C);

  printf ("  %5d    %9.4f\n", N, wctime);

}

void master(int N, double *A, double *B, double *C){}
void worker(int N, double *A, double *B, double *C){}

