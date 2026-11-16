#include "utils_1d.h"
#include <mpi.h>

int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double dx = 1.0/(num_pts+1);
  double dt = 0.5*dx*dx;

  // quanti punti ha ogni processo
  int *counts = (int *) malloc(size * sizeof(int));
  int *displs = (int *) malloc(size*sizeof(int));
  size_t quota = num_pts / size;
  size_t resto = num_pts % size;
  int tot = 0;

  for (int r = 0; r < size; r++){
    counts[r] = quota + ((size_t) r < resto ? 1 : 0);
    displs[r] = tot;
    tot += counts[r];
  }

  int mioNum = counts[rank];
  size_t inizio = 1 + displs[rank];
  size_t fine = inizio + mioNum - 1;

  int sx = MPI_PROC_NULL, dx_rank = MPI_PROC_NULL;
  if (rank > 0 && counts[rank-1] > 0) sx = rank - 1;
  if (rank < size-1 && counts[rank+1] > 0) dx_rank = rank + 1;

  double *aux = (double *) malloc((num_pts + 2) * sizeof(double));
  double localErr = 0.0;

  heat_array[0] = 0.0;
  heat_array[num_pts+1] = 0.0;
  aux[0] = 0.0;
  aux[num_pts+1] = 0.0;

  for (size_t i = 0; i < ITER; i++){

    if (mioNum > 0){
      // scambio i bordi con i vicini
      MPI_Sendrecv(&heat_array[inizio], 1, MPI_DOUBLE, sx, 0,
                   &heat_array[inizio-1], 1, MPI_DOUBLE, sx, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(&heat_array[fine], 1, MPI_DOUBLE, dx_rank, 1,
                   &heat_array[fine+1], 1, MPI_DOUBLE, dx_rank, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (size_t j = inizio; j <= fine; j++){
        aux[j] = heat_array[j] + dt/(dx*dx) * (heat_array[j-1] - 2.0*heat_array[j] + heat_array[j+1]);
        localErr += dx*fabs(aux[j] - heat_array[j]);
      }

      for (size_t j = inizio; j <= fine; j++) heat_array[j] = aux[j];
    }
  }

  double totErr = 0.0;
  MPI_Reduce(&localErr, &totErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // raccolgo tutto su rank 0
  if (rank == 0){
    *err += totErr;
    MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, heat_array+1, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else{
    MPI_Gatherv(&heat_array[inizio], mioNum, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

 
  return 0;
}
