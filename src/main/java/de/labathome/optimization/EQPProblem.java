package de.labathome.optimization;

import org.netlib.util.intW;

import com.github.fommil.netlib.LAPACK;

/**
 * Solve the Equality-Constrained Quadratic Programming Problem:
 *
 * <pre>
 * min wrt. x: q(x) = 1/2 x^T G x + x^T c
 * s.t.        Ax = b
 * </pre>
 *
 * where
 * G is the symmetric (n x n) Hessian matrix,
 * c and x are vectors in R^n and
 * A is the (m x n) Jacobian of constraints (with m <= n).
 * n : number of parameters
 * m : number of constraints
 *
 * See Nocedal & Wright, Numerical Optimization (2006), chapter 16.1
 * See also: https://antonior92.github.io/posts/2017/05/projected-CG/
 */
public class EQPProblem {

	private static final LAPACK lapack = LAPACK.getInstance();

	/** number of parameters of the problem */
	protected int n;

	/** number of constraints of the problem */
	protected int m;

	/** [n][n] Hessian matrix of the EQP problem */
	protected double[][] H;

	/** [n] gradient of the quadratic objective function */
	protected double[] c;

	/** [m][n] Jacobian matrix of the EQP problem */
	protected double[][] A;

	/** [m] Right-hand side of the constraint equation * (-1) */
	protected double[] b;

	/** [n] solution: vector of parameters */
	protected double[] x;

	/** [m] solution: Lagrange multipliers for constriants */
	protected double[] lambda;

	/**
	 * Setup an equality-constrained quadratic programming problem.
	 *
	 * @param n number of parameters of the problem
	 * @param m number of constraints of the problem
	 * @param H [n][n] Hessian matrix of the EQP problem
	 * @param c [n] gradient of the quadratic objective function
	 * @param A [m][n] Jacobian matrix of the EQP problem
	 * @param b [m] Right-hand side of the constraint equation * (-1)
	 */
	public EQPProblem(int n, int m, double[][] H, double[] c, double[][] A, double[] b) {
		this.n = n;
		this.m = m;
		this.H = H;
		this.c = c;
		this.A = A;
		this.b = b;

		x = new double[n];
		lambda = new double[m];
	}

	/** solve by direct factorization of the KKT matrix (16.5) */
	public void directFactorization() {

		// 1. build explicit KKT matrix:
		// [ G A^T ]
		// [ A  0  ]
		// It is stored in column-major format for compatibility with LAPACK.
		final double[] kkt = new double[(n+m)*(n+m)];

		// copy G into top left block of KKT matrix
		for (int iR = 0; iR < n; ++iR) {
			for (int iC=0; iC < n; ++iC) {
				kkt[iC * (n+m) + iR] = H[iR][iC];
			}
		}

		for (int j=0; j<m; ++j) {
			for (int i=0; i<n; ++i) {
				// copy A into bottom left block of KKT matrix
				kkt[i * (n+m) + (n+j)] = A[j][i];

				// copy A^T into top right block of KKT matrix
				kkt[(n+j) * (n+m) + i] = A[j][i];
			}
		}

		// 2. build RHS vector
		// [ -c ]
		// [ -b ]
		final double[] rhs = new double[n+m];
		for (int i=0; i<n; ++i) {
			rhs[i] = -c[i];
		}
		for (int j=0; j<m; ++j) {
			rhs[n+j] = -b[j]; // TODO: change to b --> also in API!
		}

		// 3. obtain LU factorization of KKT matrix using LAPACK's dgetrf
		final int[] ipiv = new int[n+m];
		intW info = new intW(0);
		lapack.dgetrf(n+m, n+m, kkt, n+m, ipiv, info);
		if (info.val != 0) {
			throw new RuntimeException(String.format("DGETRF returned info = %d", info.val));
		}

		// 4. solve using LAPACK's dgetrs
		String trans = "N";
		lapack.dgetrs(trans, n+m, 1, kkt, n+m, ipiv, rhs, n+m, info);
		if (info.val != 0) {
			throw new RuntimeException(String.format("DGETRS returned info = %d", info.val));
		}

		// 5. copy solution back into appropriate vectors
		for (int i=0; i<n; ++i) {
			x[i] = rhs[i];
		}
		for (int j=0; j<m; ++j) {
			lambda[j] = -rhs[n+j];
		}
	}

	/**
	 * Get the solution vector.
	 *
	 * @return [n] values of parameters
	 */
	public double[] getX() {
		return x;
	}

	/**
	 * Get the vector of Lagrange multipliers for the constraints.
	 *
	 * @return [m] values of Lagrange multipliers
	 */
	public double[] getLambda() {
		return lambda;
	}

}
