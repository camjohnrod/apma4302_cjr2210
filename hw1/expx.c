#include <petsc.h>
#include <petscdt.h>
#include <math.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank, num_processes, terms_per_rank;
  PetscInt       N = 3, i, num_remainder;
  PetscReal      x = 1.0, localval, globalsum, factorial_val, true_val, rel_error;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute exp(x) in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &num_processes));

  // read option
  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx","");
  PetscCall(PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL));
  PetscCall(PetscOptionsInt("-N","number of Taylor series terms to use",NULL,N,&N,NULL));
  PetscOptionsEnd();

  // compute e^x using N Taylor series terms (Horner's Method)
  terms_per_rank = N / num_processes;
  num_remainder = N - terms_per_rank * num_processes;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Taylor series terms per rank: %d\n", terms_per_rank));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of remaining terms: %d\n", num_remainder));

  localval = 0.0;
  for (i = terms_per_rank - 1; i >= 0; i--) {
    PetscDTFactorial(terms_per_rank * rank + i, &factorial_val);
    localval += 1 / factorial_val;
    if (i != 0) {
        localval *= x;   
    }
  }
    
  localval *= PetscPowReal(x, (terms_per_rank * rank));

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate and report on work from each process
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "exp(%17.15f) is about %17.15f\n",x,globalsum));

  true_val = exp(x);
  rel_error = PetscAbsReal(((true_val - globalsum) / true_val)) / PETSC_MACHINE_EPSILON;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "The relative error compared to the exp(x) function is %.5f * epsilon\n", rel_error));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}