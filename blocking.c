#include "timing.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include "mpi.h"

#define ROOT 0 // root process id
#define TAG 99
#define VERIFY 0 // should we verify our results?
#define NUM_RUNS 5
#define MIN(a,b) (((a)<(b))?(a):(b))
#define EVEN(rank) ((rank % 2) == 0)
#define ODD(rank) ((rank % 2) == 1)

// don't make this a macro b/c side effects
int start_index(int N, int i){
  return (i*(i+1)/2);
}

void debug_err(int rank, int error_code){
  if (error_code != MPI_SUCCESS) {
   char error_string[BUFSIZ];
   int length_of_error_string;
   MPI_Error_string(error_code, error_string, &length_of_error_string);
   fprintf(stderr, "%3d: %s\n", rank, error_string);
  } 
}

double buffer_init(int blockwidth, int N, double **Ablock, double **Bblock, double **B2block, double **Cblock){
  *Ablock = (double *) calloc(blockwidth * N, sizeof(double)); /* there is some extra space here */
  *Bblock = (double *) calloc(blockwidth * N, sizeof(double));
  *B2block = (double *) calloc(blockwidth * N, sizeof(double));
  *Cblock = (double *) calloc(blockwidth * N, sizeof(double));
}

void matrix_init(int, double**, double**, double**);

void master(int, int, double*, double*, double*, double*, double*, double*, double*); /* do master work and report time */

void worker(int, int, int, double*, double*, double*, double*); /* do worker work, but do not send to master */

double matmul(int, double*, double*, double*);

int main(int argc, char **argv) {

  /*
    This is the serial main program for CPSC424/524 Assignment #3.

    Author: Andrew Sherman, Yale University

    Date: 2/01/2016

  */

  int N, i, j, k, run;
  double *A, *B, *C, *C2; /* master buffers */
  double *Ablock, *Bblock, *B2block, *Cblock; /* worker buffers */
  int size, rank, sizeAB, sizeC, iA, iB, iC;

  // make sure to change NUM_RUNS along with this!
  int sizes[NUM_RUNS]={1000,2000,4000,8000,12000};
  
  double wctime0, wctime1, cputime;
  
  MPI_Status status;

  MPI_Init(&argc,&argv); // Required MPI initialization call

  MPI_Comm_size(MPI_COMM_WORLD,&size); // Get no. of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Which process am I?
  
  if (rank == ROOT) { /* master */
    printf("Timing breakdowns:\n");
    printf("Matrix multiplication times:\n   N      TIME (secs)\n\tP\tcomp\tcomm\n -----   -------------\n");

    // do all master runs
    for (run=0; run<NUM_RUNS; run++) {
      N = sizes[run];
      matrix_init(N, &A, &B, &C);
      buffer_init(N / size, N, &Ablock, &Bblock, &B2block, &Cblock);

      timing(&wctime0, &cputime);
      master(size, N, A, B, C, Ablock, Bblock, B2block, Cblock);
      timing(&wctime1, &cputime);
      
      printf ("  %5d    %9.4f\n", N, wctime1 - wctime0);

      // verify correctness
      if(VERIFY){
        printf("Verifying correctness\n");
        C2 = (double *) calloc(N*N, sizeof(double));
        matmul(N, A, B, C2);
        for(i = 0; i < N * N; i++){
          if(C[i] != C2[i]){
            printf("ERROR: element number %d is %f but should be %f\n", i, C[i], C2[i]);
          }
        }
        printf("They all match up! Hooray!\n");
      }
    }
cleanup:
    free(A); free(Ablock);
    free(B); free(Bblock);
    free(C); free(Cblock);
    free(B2block);
  
  } else { /* worker */
    for (run=0; run<NUM_RUNS; run++) {
      // init buffers
      N = sizes[run];
      buffer_init(N / size, N, &Ablock, &Bblock, &B2block, &Cblock);
      
      // wait to start
      worker(rank, size, N, Ablock, Bblock, B2block, Cblock);
  
      // gather up all the blocks into C matrix
      MPI_Gather(Cblock, (N * N)/size, MPI_DOUBLE,
          C, (N * N)/size, MPI_DOUBLE,
          ROOT, MPI_COMM_WORLD);

      // cleanup
      free(Ablock); 
      free(Bblock); free(B2block);
      free(Cblock); 
    }
  }
  
  MPI_Finalize(); // Required MPI termination call
}

void matrix_init(int N, double **A, double **B, double **C){
  int sizeAB, sizeC, i;

  sizeAB = N*(N+1)/2; //Only enough space for the nonzero portions of the matrices
  sizeC = N*N; // All of C will be nonzero, in general!

  *A = (double *) calloc(sizeAB, sizeof(double));
  *B = (double *) calloc(sizeAB, sizeof(double));
  *C = (double *) calloc(sizeC, sizeof(double));

  srand(12345); // Use a standard seed value for reproducibility

  // This assumes A is stored by rows, and B is stored by columns. Other storage schemes are permitted
  for (i=0; i<sizeAB; i++) (*A)[i] = ((double) rand()/(double)RAND_MAX);
  for (i=0; i<sizeAB; i++) (*B)[i] = ((double) rand()/(double)RAND_MAX);
}

void master(int p, int N, double *A, double *B, double *C, double *Ablock, double *Bblock, double *B2block, double *Cblock){
  int i;
  int buf_offset, buf_len, block_width;

  // how many rows does each processor get?
  block_width = N / p;
  
  // assign row/column blocks
  for(i = 0; i < p; i++){
    buf_offset = start_index(N, i * block_width);
    buf_len = start_index(N, (i + 1) * block_width) - buf_offset;


    // assign A block
    MPI_Send(&buf_offset, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(&buf_len, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(A + buf_offset, buf_len, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
    
    // assign B block
    MPI_Send(&buf_offset, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(&buf_len, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
    MPI_Send(B + buf_offset, buf_len, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
  }
  
  // run as a worker
  worker(ROOT, p, N, Ablock, Bblock, B2block, Cblock);

  // gather up all the blocks into C matrix
  MPI_Gather(Cblock, (N * N)/p, MPI_DOUBLE,
      C, (N * N)/p, MPI_DOUBLE,
      ROOT, MPI_COMM_WORLD);
}

void worker(int rank, int p, int N, double *Ablock, double *Bblock, double *B2block, double *Cblock){
  int round, next, block_idx, i, j, k, iA, iA_len, iB, iB_len, iC;
  int A_buf_offset, A_buf_len, block_width;
  int B_buf_offset, B_buf_len;
  int B2_buf_offset, B2_buf_len;
  double *tmp; // for buffer swapping
  double wctime0, wctime1, cputime, comp_time, comm_time;
  MPI_Status status;
  
  block_width = N / p;
  comp_time = 0.0;
  comm_time = 0.0;

  // get row assignment
  timing(&wctime0, &cputime);
  MPI_Recv(&A_buf_offset, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(&A_buf_len, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(Ablock, A_buf_len, MPI_DOUBLE, ROOT, TAG, MPI_COMM_WORLD, &status);
  timing(&wctime1, &cputime);
  comm_time += wctime1 - wctime0;

  for(round = 0; round < p; round++){
    // which B column are we dealing with?
    block_idx = (rank + round) % p;

    if(EVEN(rank) || round == 0){
      // receive a row
      timing(&wctime0, &cputime);
      MPI_Recv(&B_buf_offset, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&B_buf_len, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(Bblock, B_buf_len, MPI_DOUBLE, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      timing(&wctime1, &cputime);
      comm_time += wctime1 - wctime0;
    } else {
      // switch the buffers since we already have it!
      tmp = B2block;
      B2block = Bblock;

      B_buf_offset = B2_buf_offset;
      B_buf_len = B2_buf_len;

      Bblock = tmp;
    }
    
    if(ODD(rank) && round != p - 1){
      // receive a row
      timing(&wctime0, &cputime);
      MPI_Recv(&B2_buf_offset, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&B2_buf_len, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(B2block, B2_buf_len, MPI_DOUBLE, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
      timing(&wctime1, &cputime);
      comm_time += wctime1 - wctime0;
    }
    
    // if(round == 0) MPI_Barrier(MPI_COMM_WORLD); // halt to prevent overlap

    // Processing loop 
    timing(&wctime0, &cputime);
    for(i = 0; i < block_width; i++){
      iA = start_index(N, rank * block_width + i) - A_buf_offset;
      for(j = 0; j < block_width; j++){
        iB = start_index(N, block_idx * block_width + j) - B_buf_offset;
        iC = (N * i) + block_idx * block_width + j;
        Cblock[iC] = 0.;
        
        for (k = 0; k <= MIN(rank * block_width + i, block_idx * block_width + j); k++){
          Cblock[iC] += Ablock[iA+k] * Bblock[iB+k];
        }
      }
    }
    timing(&wctime1, &cputime);
    comp_time += wctime1 - wctime0;

    // pass data to the left
    if(round != p - 1){
      next = (rank + p - 1) % p;

      // if we're an odd processor, we'll receive first (MUST be odd ones)
      timing(&wctime0, &cputime);
      MPI_Send(&B_buf_offset, 1, MPI_INT, next, TAG, MPI_COMM_WORLD);
      MPI_Send(&B_buf_len, 1, MPI_INT, next, TAG, MPI_COMM_WORLD);
      MPI_Send(Bblock, B_buf_len, MPI_DOUBLE, next, TAG, MPI_COMM_WORLD);
      timing(&wctime1, &cputime);
      comm_time += wctime1 - wctime0;
    }
  }
  printf("\t%d\t%f\t%f\n", rank, comp_time, comm_time);
}

