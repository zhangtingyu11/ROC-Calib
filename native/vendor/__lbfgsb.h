#pragma once
#include <math.h>
// BLAS
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void dscal_(int* n, double* alpha, double* x, int* incx);
void dcopy_(int* n, double* x, int* incx, double* y, int* incy);
double dnrm2_(int* n, double* x, int* incx);
double ddot_(int* n, double* x, int* incx, double* y, int* incy);

// LAPACK
void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
void dtrtrs_(char* uplo, char* trans, char* diag, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, int* info);

void setulb(int n, int m, double* x, double* l, double* u, int* nbd, double* f,
       double* g, double factr, double pgtol, double* wa, int* iwa, int* task,
       int* lsave, int* isave, double* dsave, int maxls, int* ln_task
);

