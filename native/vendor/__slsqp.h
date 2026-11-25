/*
 * This file and the accompanying __slsqp.c file are the C translations of the
 * Fortran77 code of the SLSQP algorithm for the SciPy project and hence inherits
 * SciPy license. The original Fortran code is available at
 * http://www.netlib.org/toms/733 written by Dieter Kraft, see:
 *
 *  ALGORITHM 733, COLLECTED ALGORITHMS FROM ACM.
 *  TRANSACTIONS ON MATHEMATICAL SOFTWARE,
 *  VOL. 20, NO. 3, SEPTEMBER, 1994, PP. 262-281.
 *  https://doi.org/10.1145/192115.192124
 *
 *
 * The original Fortran code is released for use under BSD license, with the
 * following statement from the original license holder ACM publications:
  *
 *  https://web.archive.org/web/20170106155705/http://permalink.gmane.org/gmane.comp.python.scientific.devel/6725
 *  ------
 *  From: Deborah Cotton <cotton@hq.acm.org>
 *  Date: Fri, 14 Sep 2007 12:35:55 -0500
 *  Subject: RE: Algorithm License requested
 *  To: Alan Isaac
 *
 *  Prof. Issac,
 *
 *  In that case, then because the author consents to [the ACM] releasing
 *  the code currently archived at http://www.netlib.org/toms/733 under the
 *  BSD license, the ACM hereby releases this code under the BSD license.
 *
 *  Regards,
 *
 *  Deborah Cotton, Copyright & Permissions
 *  ACM Publications
 *  2 Penn Plaza, Suite 701**
 *  New York, NY 10121-0701
 *  permissions@acm.org
 *  212.869.7440 ext. 652
 *  Fax. 212.869.0481
 *  ------
*/

#ifndef __SLSQPLIB_H
#define __SLSQPLIB_H






#include <math.h>
#include "__nnls.h"

// BLAS/LAPACK function prototypes used in SLSQP
void daxpy_(int* n, double* sa, double* sx, int* incx, double* sy, int* incy);
double ddot_(int* n, double* dx, int* incx, double* dy, int* incy);
void dgelsy_(int* m, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, int* jpvt, double* rcond, int* rank, double* work, int* lwork, int* info);
void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
void dgeqr2_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* info);
void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, double* lwork, int* info);
void dgerq2_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* info);
void dlarf_(char* side, int* m, int* n, double* v, int* incv, double* tau, double* c, int* ldc, double* work);
void dlarfgp_(int* n, double* alpha, double* x, int* incx, double* tau);
void dlartgp_(double* f, double* g, double* cs, double* sn, double* r);
double dnrm2_(int* n, double* x, int* incx);
void dorm2r_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda, double* tau, double* c, int* ldc, double* work, int* info);
void dormr2_(char* side, char* trans, int* m, int* n, int* k, double* a, int* lda, double* tau, double* c, int* ldc, double* work, int* info);
void dscal_(int* n, double* da, double* dx, int* incx);
void dtpmv_(char* uplo, char* trans, char* diag, int* n, double* ap, double* x, int* incx);
void dtpsv_(char* uplo, char* trans, char* diag, int* n, double* ap, double* x, int* incx);
void dtrsm_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, double* alpha, double* a, int* lda, double* b, int* ldb);
void dtrsv_(char* uplo, char* trans, char* diag, int* n, double* a, int* lda, double* x, int* incx);


// The SLSQP_vars struct holds the state of the algorithm and passed to Python
// and back such that it is thread-safe.
struct SLSQP_vars {
    double acc, alpha, f0, gs, h1, h2, h3, h4, t, t0, tol;
    int exact, inconsistent, reset, iter, itermax, line, m, meq, mode, n;
};


void __slsqp_body(struct SLSQP_vars* S, double* funx, double* gradx, double* C, double* d, double* sol, double* mult, double* xl, double* xu, double* buffer, int* indices);



#endif
