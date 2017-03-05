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

int buf_offset (int N, int p, int column){
  int block_width = N / p;
  return start_index(N, column * block_width);
}

int buf_len (int N, int p, int column){
  int block_width = N / p;
  return start_index(N, (column + 1) * block_width) - buf_offset(N, p, column);
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

  if(!Ablock || !Bblock || !Cblock || !B2block)
    printf("Calloc failed!\n");
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
  int sizes[NUM_RUNS]={1000,12000,4000,8000,12000};
  //int sizes[NUM_RUNS]={1000};
  
  double wctime0, wctime1, cputime;
  
  MPI_Status status;

  MPI_Init(&argc,&argv); // Required MPI initialization call

  MPI_Comm_size(MPI_COMM_WORLD,&size); // Get no. of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Which process am I?
  
  if (rank == ROOT) { /* master */
    printf("Matrix multiplication times:\n   N      TIME (secs)\n -----   -------------\n");

    // do all master runs
    for (run=0; run<NUM_RUNS; run++) {
      MPI_Barrier(MPI_COMM_WORLD);
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
          if((float)C[i] != (float)C2[i]){
            printf("ERROR: element number %d is %f but should be %f\n", i, C[i], C2[i]);
          }
        }
        printf("They all match up! Hooray!\n");
        free(C2);
      }
cleanup:
      free(A); free(Ablock);
      free(B); free(Bblock);
      free(C); free(Cblock);
      free(B2block);
    }
  } else { /* worker */
    for (run=0; run<NUM_RUNS; run++) {
      MPI_Barrier(MPI_COMM_WORLD);
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

  // printf("COLUMNS!\n");
  // for(int col = 0; col < N; col++){
  //   for (int x = start_index(N, col); x < start_index(N, col + 1); x++){
  //     printf("%f ", (*B)[x]);
  //   }
  //   printf("\n");
  // }
}

void master(int p, int N, double *A, double *B, double *C, double *Ablock, double *Bblock, double *B2block, double *Cblock){
  int i;
  int buf_offset, buf_len, block_width;
  MPI_Request req;

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
    // printf("\tMaster assigning process %d column with: offset = %d, len = %d\n", i, buf_offset, buf_len);
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
  int B2_buf_offset, B2_buf_len, tmp_offset, tmp_len;
  double *tmp; // for buffer swapping
  MPI_Status status;
  MPI_Request send1, send2, recv, tmp_req; // primary and secondary requests
  
  block_width = N / p;

  // get row assignment
  MPI_Recv(&A_buf_offset, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(&A_buf_len, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(Ablock, A_buf_len, MPI_DOUBLE, ROOT, TAG, MPI_COMM_WORLD, &status);
  
  // get initial row
  MPI_Recv(&B_buf_offset, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
  MPI_Recv(&B_buf_len, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
  // printf("Process %d getting column assignment, offset = %d, len = %d\n", rank, B_buf_offset, B_buf_len);
  MPI_Recv(Bblock, B_buf_len, MPI_DOUBLE, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);

  // wait to start
  MPI_Barrier(MPI_COMM_WORLD);

  for(round = 0; round < p; round++){
    // which B column are we dealing with?
    block_idx = (rank + round) % p;

    if(round == 0 && round != p - 1){
      // TODO start sending the data we have
      // and start receiving next data
      next = (rank + p - 1) % p;
      MPI_Isend(&B_buf_offset, 1, MPI_INT, next, TAG, MPI_COMM_WORLD, &send1);
      MPI_Isend(&B_buf_len, 1, MPI_INT, next, TAG, MPI_COMM_WORLD, &send1);
      MPI_Isend(Bblock, N * block_width, MPI_DOUBLE, next, TAG, MPI_COMM_WORLD, &send1);

      MPI_Irecv(&B2_buf_offset, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
      MPI_Irecv(&B2_buf_len, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
      MPI_Irecv(B2block, block_width * N, MPI_DOUBLE, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
    }

    // Processing loop 
    // printf("Process %d on round %d has the following column:", rank, round);
    // for(int x = 0; x < B_buf_len; x++) printf(" %f", Bblock[x]);
    // printf("\n");
    for(i = 0; i < block_width; i++){
      iA = start_index(N, rank * block_width + i) - A_buf_offset;
      for(j = 0; j < block_width; j++){
        iB = start_index(N, block_idx * block_width + j) - B_buf_offset;
        iC = (N * i) + block_idx * block_width + j;
        Cblock[iC] = 0.;
        
        for (k = 0; k <= MIN(rank * block_width + i, block_idx * block_width + j); k++){
          Cblock[iC] += Ablock[iA+k] * Bblock[iB+k];
          // printf("Processor %d on round %d calculating Cblock[%d] = Ablock[%d+%d] * Bblock[%d+%d] = %f, products are %f and %f\n",
          //     rank, round, iC, iA, k, iB, k, Cblock[iC], Ablock[iA+k], Bblock[iB+k]);
        }
      }
    }

    if(round != p - 1){
      // finish receiving next column
      next = (rank + p - 1) % p;
      MPI_Wait(&recv, &status);
      // printf("Process %d received a column from %d with length %d, offset %d\n", rank, rank + 1 % p, B2_buf_len, B2_buf_offset);

      if(round != p - 2){
        MPI_Isend(&B2_buf_offset, 1, MPI_INT, next, TAG, MPI_COMM_WORLD, &send2);
        MPI_Isend(&B2_buf_len, 1, MPI_INT, next, TAG, MPI_COMM_WORLD, &send2);
        MPI_Isend(B2block, N * block_width, MPI_DOUBLE, next, TAG, MPI_COMM_WORLD, &send2);
        // printf("Process %d sent %d a column of buf length %d, offset %d\n", rank, next, B2_buf_len, B2_buf_offset);
        
        MPI_Wait(&send1, &status);
      }

      // now switch buffers and requests
      tmp = B2block;
      tmp_req = send2;
      tmp_offset = B2_buf_offset;
      tmp_len = B2_buf_len;

      B2block = Bblock;
      send2 = send1;
      B2_buf_offset = B_buf_offset;
      B2_buf_len = B_buf_len;

      B_buf_offset = tmp_offset;
      B_buf_len = tmp_len;
      Bblock = tmp;
      send1 = tmp_req;

      // now start receiving next round
      if(round != p - 2){
        MPI_Irecv(&B2_buf_offset, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
        MPI_Irecv(&B2_buf_len, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
        MPI_Irecv(B2block, block_width * N, MPI_DOUBLE, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &recv);
      }
    }
  }
}

