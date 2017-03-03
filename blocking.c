#include "timing.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include "mpi.h"

#define ROOT 0 // root process id
#define TAG 99
#define VERIFY 1 // should we verify our results?
#define NUM_RUNS 5
#define MIN(a,b) (((a)<(b))?(a):(b))

// don't make this a macro b/c side effects
int start_index(int N, int i){
  return (i*(i+1)/2);
}

double buffer_init(int blockwidth, int N, double *Ablock, double *Bblock, double *Cblock){
  Ablock = (double *) calloc(blockwidth * N, sizeof(double)); /* there is some extra space here */
  Bblock = (double *) calloc(blockwidth * N, sizeof(double));
  Cblock = (double *) calloc(blockwidth * N, sizeof(double));
}

double matrix_init(int, double*, double*, double*);

double master(int, double*, double*, double*, double*, double*, double*); /* do master work and report time */

void worker(int, int, double*, double*, double*); /* do worker work, but do not send to master */

double matmul(int, double*, double*, double*);

int main(int argc, char **argv) {

  /*
    This is the serial main program for CPSC424/524 Assignment #3.

    Author: Andrew Sherman, Yale University

    Date: 2/01/2016

  */

  int N, i, j, k, run;
  double *A, *B, *C, *C2; /* master buffers */
  double *Ablock, *Bblock, *Cblock; /* worker buffers */
  int sizeAB, sizeC, iA, iB, iC;

  // make sure to change NUM_RUNS along with this!
  int sizes[NUM_RUNS]={1000,2000,4000,8000,12000};

  double wctime0, wctime1, cputime;
  
  MPI_Status status;

  MPI_Init(&argc,&argv); // Required MPI initialization call

  MPI_Comm_size(MPI_COMM_WORLD,&size); // Get no. of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Which process am I?
  
  if (rank == ROOT) { /* master */
    printf("Matrix multiplication times:\n   N      TIME (secs)\n -----   -------------\n");

    // do all master runs
    for (run=0; run<NUM_RUNS; run++) {
      N = sizes[run];
      matrix_init(N, A, B, C);
      buffer_init(N / size, N, Ablock, Bblock, Cblock);
      MPI_Barrier(MPI_COMM_WORLD);

      timing(&wctime0, &cputime);
      master(N, A, B, C);
      timing(&wctime1, &cputime);
      
      printf ("  %5d    %9.4f\n", N, wctime1 - wctime0);

      // verify correctness
      if(VERIFY){
        matmul(N, A, B, C2);
        for(i = 0; i < N * N; i++){
          if(C[i] != C2[i]){
            printf("ERROR: element number %d is %f but should be %f\n", i, C[i], C2[i]);
            goto cleanup;
          }
        }
        printf("They all match up! Hooray!\n");
      }
    }
cleanup:
    free(A); free(Ablock);
    free(B); free(Bblock);
    free(C); free(Cblock);
  
  } else { /* worker */
    for (run=0; run<NUM_RUNS; run++) {
      // init buffers
      N = sizes[run];
      buffer_init(N / size, N, Ablock, Bblock, Cblock);
      
      // wait to start
      MPI_Barrier(MPI_COMM_WORLD);
      worker(N, A, B, C);
  
      // gather up all the blocks into C matrix
      MPI_Gather(Cblock, (N * N)/p, MPI_DOUBLE,
          C, (N * N)/p, MPI_DOUBLE,
          ROOT, MPI_COMM_WORLD);

      // cleanup
      free(Ablock); 
      free(Bblock); 
      free(Cblock); 
    }
  }
  
  MPI_Finalize(); // Required MPI termination call
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
}

void master(int p, int N, double *A, double *B, double *C, double *Ablock, double *Bblock, double *Cblock){
  // TODO
  //
  //  (1) Assign a row block of A to each processor
  //  (2) Assign a column block of B to each processor
  //  (3) Run Worker
  //  (4) Collect results using gather
  int i;
  int buf_offset, buf_len, block_width;
  struct rowblock data;

  // how many rows does each processor get?
  block_width = N / p;
  
  // assign row/column blocks
  for(i = 0; i < p; i++){
    buf_offset = start_index(N, i * block_width);
    buf_len = start_index(N, (i + 1) * block_width) - buf_offset;

    // assign A block
    MPI_Send(&buf_offset, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(&buf_len, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(A, buf_len, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
    
    // assign B block
    MPI_Send(&buf_offset, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(&buf_len, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(B, buf_len, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
  }

  // run as a worker
  worker(ROOT, p, N, Ablock, Bblock, Cblock);

  // gather up all the blocks into C matrix
  MPI_Gather(Cblock, (N * N)/p, MPI_DOUBLE,
      C, (N * N)/p, MPI_DOUBLE,
      ROOT, MPI_COMM_WORLD);
}

void worker(int rank, int p, int N, double *Ablock, double *Bblock, double *Cblock){
  // TODO
  //
  //  (1) Wait for row assignment
  //  (2) For i = 0 --> num_procs
  //      - Wait for column "j" assignment
  //      - Calculate j'th block of C
  //  (3) Send C block row to master
  //
  int round, block_idx, i, j, k, iA, iA_len, iB, iB_len iC;
  int A_buf_offset, A_buf_len, block_width;
  int B_buf_offset, B_buf_len;
  MPI_Status status;
  
  block_width = N / p;

  // get row assignment
  MPI_Recv(&A_buf_offset, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(&A_buf_len, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(Ablock, buf_len, MPI_DOUBLE, ROOT, TAG, MPI_COMM_WORLD, &status);

  for(round = 0; round < p; round++){
    // which B column are we dealing with?
    block_idx = (rank + round) % p;
    
    // receive a row
    MPI_Recv(&B_buf_offset, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&B_buf_len, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(Bblock, B_buf_len, MPI_DOUBLE, ROOT, TAG, MPI_COMM_WORLD, &status);

    if(round == 0) MPI_Barrier(MPI_COMM_WORLD); // halt to prevent overlap

    // Processing loop 
    for(i = 0; i < block_width; i++){
      iA = start_index(N, rank * block_width + i) - A_buf_offset;
      for(j = 0; j < block_width; j++){
        iB = start_index(N, block_idx * block_width + j) - B_buf_offset;
        iC = (N * i) + j;
        Cblock[iC] = 0.;
        for (k = 0; k <= MIN(rank * block_width + i, block_idx * block_width + j); k++) Cblock[iC] += Ablock[iA+k] + Bblock[iB+k];
      }
    }
  }
}

