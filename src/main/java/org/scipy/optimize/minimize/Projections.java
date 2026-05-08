package org.scipy.optimize.minimize;

import java.util.Arrays;

import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.DenseSolve;
import org.scipy.optimize.minimize.sparse.SparseAssembly;
import org.scipy.optimize.minimize.sparse.UjmpBridge;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Chol.CholMatrix;
import org.ujmp.core.doublematrix.calculation.general.decomposition.QR.QRMatrix;
import org.ujmp.core.doublematrix.calculation.general.decomposition.SVD.SVDMatrix;

public class Projections {

	/**
	 * Measure orthogonality between a vector and the null space of a matrix.
	 *
	 * Compute a measure of orthogonality between the null space
	 * of the (possibly sparse) matrix {@code A} and a given vector {@code g}.
	 *
	 * The formula is a simplified (and cheaper) version of formula (3.13) from [1]:
	 * {@code orth =  norm(A g, ord=2)/(norm(A, ord='fro')*norm(g, ord=2))}
	 *
	 * [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
	 *     "On the solution of equality constrained quadratic
	 *      programming problems arising in optimization."
	 *      SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
	 *
	 * @param A [n][m] matrix
	 * @param g [m] vector
	 * @return how orthogonal g is to the nullspace of A
	 */
	public static double orthogonality(Matrix A, Matrix g) {
		if (!g.isRowVector()) {
			throw new RuntimeException("g has to be a (n x 1)-vector");
		}
		if (A.getColumnCount() != g.getRowCount()) {
			throw new RuntimeException("cols(A) must be equal to rows(g)");
		}

		double normG = g.norm2();
		double normA = A.normF();

		// Check if norms are zero
		if (normG == 0.0 || normA == 0.0) {
			return 0.0;
		}

		double normAg = A.mtimes(g).norm2();

		// Orthogonality measure
		double orth = normAg/(normA * normG);

		return orth;
	}

	public static LinearOperator[] projections(Matrix A)  {
		ProjectionMethod method = null;
		return projections(A, method);
	}

	public static LinearOperator[] projections(Matrix A, ProjectionMethod method)  {
		double orthTol = 1.0e-12;
		return projections(A, method, orthTol);
	}

	public static LinearOperator[] projections(Matrix A, ProjectionMethod method, double orthTol)  {
		int maxRefine = 3;
		return projections(A, method, orthTol, maxRefine);
	}

	public static LinearOperator[] projections(Matrix A, ProjectionMethod method, double orthTol, int maxRefine)  {
		double tolerance = 1.0e-15;
		return projections(A, method, orthTol, maxRefine, tolerance);
	}

	/**
	 * Return three linear operators related with a given matrix A.
	 *
	 * Uses iterative refinements described in [1]
	 * during the computation of {@code Z} in order to
	 * cope with the possibility of large roundoff errors.
	 *
	 * [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
	 *     "On the solution of equality constrained quadratic
	 *     programming problems arising in optimization."
	 *     SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
	 *
	 * @param A [m][n] Matrix {@code A} used in the projection.
	 * @param method Method used for compute the given linear operators.
	 * @param orthTol Tolerance for iterative refinements.
	 * @param maxRefine Maximum number of iterative refinements.
	 * @param tolerance Tolerance for singular values.
	 * @return [3]: {Z, LS, Y} with
	 *              Z : LinearOperator, shape (n, n)
	 *                  Null-space operator. For a given vector {@code x},
	 *                  the null space operator is equivalent to apply
	 *                  a projection matrix {@code P = I - A.T inv(A A.T) A}
	 *                  to the vector. It can be shown that this is
	 *                  equivalent to project {@code x} into the null space
	 *                  of {@code A}.
	 *              LS : LinearOperator, shape (m, n)
	 *                  Least-squares operator. For a given vector {@code x},
	 *                  the least-squares operator is equivalent to apply a
	 *                  pseudoinverse matrix {@code pinv(A.T) = inv(A A.T) A}
	 *                  to the vector. It can be shown that this vector
	 *                  {@code pinv(A.T) x} is the least_square solution to
	 *                  {@code A.T y = x}.
	 *              Y : LinearOperator, shape (n, m)
	 *                  Row-space operator. For a given vector {@code x},
	 *                  the row-space operator is equivalent to apply a
	 *                  projection matrix {@code Q = A.T inv(A A.T)}
	 *                  to the vector.  It can be shown that this
	 *                  vector {@code y = Q x}  the minimum norm solution
	 *                  of {@code A y = x}.
	 */
	public static LinearOperator[] projections(Matrix A, ProjectionMethod method, double orthTol, int maxRefine, double tolerance)  {

		// Pick a default method if the caller hasn't. Prefer QR for dense and
		// AugmentedSystem for sparse. Note: since the AugmentedSystem path now
		// goes through the in-tree CSR module (regardless of whether A is a
		// UJMP dense or sparse Matrix), AugmentedSystem and SVD/QR are
		// interchangeable for any A — the historical sparse-only / dense-only
		// gates from scipy don't apply to this port and are no longer
		// enforced here.
		if (method == null) {
			method = A.isSparse()
					? ProjectionMethod.AUGMENTED_SYSTEM
					: ProjectionMethod.QR_FACTORIZATION;
		}

		switch (method) {
		case NORMAL_EQUATION:
			return normalEquationProjections(A, orthTol, maxRefine, tolerance);
		case AUGMENTED_SYSTEM:
			return augmentedSystemProjections(A, orthTol, maxRefine, tolerance);
		case QR_FACTORIZATION:
			return qrFactorizationProjections(A, orthTol, maxRefine, tolerance);
		case SVD_FACTORIZATION:
			return svdFactorizationProjections(A, orthTol, maxRefine, tolerance);
		default:
			throw new RuntimeException("Method not implemented yet.");
		}
	}

	/**
	 * Return linear operators for matrix A using {@code NORMAL_EQUATION} approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return { nullSpace, leastSquares, rowSpace }
	 */
	private static LinearOperator[] normalEquationProjections(Matrix A, double orthTol, int maxRefine, double tolerance) {

		// TODO: this can be done more elegantly for sure...
		final CholMatrix cholAAt = new CholMatrix(A.mtimes(A.transpose()));

		/** z = x - A.T inv(A A.T) A x */
		LinearOperator nullSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				Matrix v = cholAAt.solve(A.mtimes(x));
				Matrix z = x.minus(A.transpose().mtimes(v));

				// Iterative refinement to improve roundoff
				// errors described in [2]_, algorithm 5.1.
				int k = 0;
				while (orthogonality(A, z) > orthTol) {
					if (k >= maxRefine) {
						break;
					}

					// z_next = z - A.T inv(A A.T) A z
					v = cholAAt.solve(A.mtimes(z));
					z = z.minus(A.transpose().mtimes(v));

					k++;
				}
				return z;
			}
		};

		/** z = inv(A A.T) A x */
		LinearOperator leastSquares = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				return cholAAt.solve(A.mtimes(x));
			}
		};

		/** z = A.T inv(A A.T) x */
		LinearOperator rowSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				return A.transpose().mtimes(cholAAt.solve(x));
			}
		};

		return new LinearOperator[] { nullSpace, leastSquares, rowSpace };
	}

	/**
	 * Return linear operators for matrix A - {@code AUGMENTED_SYSTEM}.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return { nullSpace, leastSquares, rowSpace }
	 */
	private static LinearOperator[] augmentedSystemProjections(Matrix A, double orthTol, int maxRefine, double tolerance) {

		final int n = (int) A.getColumnCount();
		final int m = (int) A.getRowCount();

		// Form augmented KKT system in CSR via the new sparse-aware block constructor:
		// [ I  A^T ]
		// [ A   0  ]
		CSRMatrix aCsr = UjmpBridge.toCSR(A);
		CSRMatrix aTCsr = aCsr.transpose().toCSR();
		final CSRMatrix kkt = SparseAssembly.blockArray(new CSRMatrix[][] {
				{ CSRMatrix.eye(n), aTCsr },
				{ aCsr,             null  },
		});

		// Factor the assembled KKT. Currently we materialise to dense and call LAPACK
		// dgetrf via dev.ludovic.netlib; when the project ships a true sparse direct
		// solver this is the single call site that needs to switch.
		// TODO: Use a symmetric indefinite factorization to solve the system twice
		//       as fast (because of the symmetry).
		final DenseSolve.LUFactor lu;
		try {
			lu = DenseSolve.factor(kkt.toDense());
		} catch (ArithmeticException ex) {
			System.out.println("Singular Jacobian matrix. Using dense SVD decomposition to \n"
					+ "perform the factorizations.");
			return svdFactorizationProjections(A, orthTol, maxRefine, tolerance);
		}

		/**
		 * z = x - A.T inv(A A.T) A x
		 * is computed solving the extended system:
		 * <pre>
		 * [I A.T] * [ z ] = [x]
		 * [A  O ]   [aux]   [0]
		 * </pre>
		 */
		LinearOperator nullSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix xMat) {
				double[] x = UjmpBridge.colToArray(xMat);
				// v = [x; 0]
				double[] v = new double[n + m];
				System.arraycopy(x, 0, v, 0, n);

				// lu_sol = [ z; aux ]
				double[] luSol = lu.solve(v);
				double[] z = Arrays.copyOfRange(luSol, 0, n);

				// Iterative refinement to improve roundoff errors
				// (Bjorck "Numerical Methods for Least Squares Problems" 1996, alg. 5.2).
				int k = 0;
				while (orthogonality(A, UjmpBridge.arrayToCol(z)) > orthTol) {
					if (k >= maxRefine) {
						break;
					}
					// new_v = v - K * lu_sol
					double[] kx = kkt.matvec(luSol);
					double[] newV = new double[n + m];
					for (int i = 0; i < n + m; ++i) {
						newV[i] = v[i] - kx[i];
					}
					double[] luUpdate = lu.solve(newV);
					for (int i = 0; i < n + m; ++i) {
						luSol[i] += luUpdate[i];
					}
					z = Arrays.copyOfRange(luSol, 0, n);
					k++;
				}
				return UjmpBridge.arrayToCol(z);
			}
		};

		/**
		 * z = inv(A A.T) A x
		 * is computed solving the extended system:
		 * [I A.T] * [aux] = [x]
		 * [A  O ]   [ z ]   [0]
		 */
		LinearOperator leastSquares = new LinearOperator() {
			@Override
			public Matrix apply(Matrix xMat) {
				double[] x = UjmpBridge.colToArray(xMat);
				// v = [x; 0]
				double[] v = new double[n + m];
				System.arraycopy(x, 0, v, 0, n);
				double[] luSol = lu.solve(v);
				return UjmpBridge.arrayToCol(Arrays.copyOfRange(luSol, n, n + m));
			}
		};

		/**
		 * z = A.T inv(A A.T) x
		 * is computed solving the extended system:
		 * [I A.T] * [ z ] = [0]
		 * [A  O ]   [aux]   [x]
		 */
		LinearOperator rowSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix xMat) {
				double[] x = UjmpBridge.colToArray(xMat);
				// v = [0; x]
				double[] v = new double[n + m];
				System.arraycopy(x, 0, v, n, m);
				double[] luSol = lu.solve(v);
				return UjmpBridge.arrayToCol(Arrays.copyOfRange(luSol, 0, n));
			}
		};

		return new LinearOperator[] { nullSpace, leastSquares, rowSpace };
	}

	/**
	 * Return linear operators for matrix A using {@code QR_FACTORIZATION} approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return { nullSpace, leastSquares, rowSpace }
	 */
	private static LinearOperator[] qrFactorizationProjections(Matrix A, double orthTol, int maxRefine, double tolerance) {

		// QR factorization of A^T
		QRMatrix qr = new QRMatrix(A.transpose());
		Matrix Q = qr.getQ();
		Matrix R = qr.getR();

		// check for inf-norm of last row in R factor:
		// if less than tolerance, use SVD factorization
		Matrix lastRowOfR = Matrix.Factory.zeros(1, R.getColumnCount());
		long rowsR = R.getRowCount();
		for (long[] pos: lastRowOfR.allCoordinates()) {
			lastRowOfR.setAsDouble(R.getAsDouble(rowsR-1, pos[1]), pos);
		}
		if (lastRowOfR.normInf() < tolerance) {
			System.out.println("Singular Jacobian matrix. Using SVD decomposition to \n" +
					           "perform the factorizations.");
			return svdFactorizationProjections(lastRowOfR, orthTol, maxRefine, tolerance);
		}

		/** z = x - A.T inv(A A.T) A x */
		LinearOperator nullSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// v = inv(R) Q.T x
				Matrix aux1 = Q.transpose().mtimes(x);
				Matrix v = R.solve(aux1);

				Matrix z = x.minus(A.transpose().mtimes(v));

				// Iterative refinement to improve roundoff
		        // errors described in [2], algorithm 5.1.
				int k = 0;
				while (orthogonality(A, z) > orthTol) {
					if (k >= maxRefine) {
						break;
					}

					//  v = inv(R) Q.T x
					aux1 = Q.transpose().mtimes(z);
					v = R.solve(aux1);

					// z_next = z - A.T v
					z = z.minus(A.transpose().mtimes(v));

					k++;
				}
				return z;
			}
		};

		/** z = inv(A A.T) A x */
		LinearOperator leastSquares = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// z = inv(R) Q.T x
				Matrix aux1 = Q.transpose().mtimes(x);
				Matrix z = R.solve(aux1);

				return z;
			}
		};

		/** z = A.T inv(A A.T) x */
		LinearOperator rowSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// z = Q inv(R.T) P.T x
				Matrix aux2 = R.transpose().solve(x);
				Matrix z = Q.mtimes(aux2);

				return z;
			}
		};

		return new LinearOperator[] { nullSpace, leastSquares, rowSpace };
	}

	/**
	 * Return linear operators for matrix A using {@code SVD_FACTORIZATION} approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return { nullSpace, leastSquares, rowSpace }
	 */
	private static LinearOperator[] svdFactorizationProjections(Matrix A, double orthTol, int maxRefine, double tolerance) {

		// SVD Factorization
		SVDMatrix svd = new SVDMatrix(A);
		Matrix U = svd.getU();
		Matrix Vt = svd.getV().transpose();
		Matrix invS = svd.getreciprocalS().transpose();

		// TODO: Remove dimensions related with very small singular values

		// z = x - A.T inv(A A.T) A x
		LinearOperator nullSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// v = U 1/s V.T x = inv(A A.T) A x
				Matrix aux1 = Vt.mtimes(x);
				Matrix aux2 = invS.mtimes(aux1);
				Matrix v = U.mtimes(aux2);
				Matrix z = x.minus(A.transpose().mtimes(v));

				// Iterative refinement to improve roundoff
		        // errors described in [2]_, algorithm 5.1.
				int k = 0;
				while (orthogonality(A, z) > orthTol) {
					if (k >= maxRefine) {
						break;
					}

					// v = U 1/s V.T x = inv(A A.T) A x
					aux1 = Vt.mtimes(z);
					aux2 = invS.mtimes(aux1);
					v = U.mtimes(aux2);

					// z_next = z - A.T v
					z = z.minus(A.transpose().mtimes(v));

					k++;
				}
				return z;
			}
		};

		/** z = inv(A A.T) A x */
		LinearOperator leastSquares = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// z = U 1/s V.T x = inv(A A.T) A x
				Matrix aux1 = Vt.mtimes(x);
				Matrix aux2 = invS.mtimes(aux1);
				Matrix z = U.mtimes(aux2);

				return z;
			}
		};

		/** z = A.T inv(A A.T) x */
		LinearOperator rowSpace = new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {

				// z = V 1/s U.T x
				Matrix aux1 = U.transpose().mtimes(x);
				Matrix aux2 = invS.transpose().mtimes(aux1);
				Matrix z = Vt.transpose().mtimes(aux2);

				return z;
			}
		};

		return new LinearOperator[] { nullSpace, leastSquares, rowSpace };
	}
}
