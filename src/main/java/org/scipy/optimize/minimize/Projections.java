package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix2D;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.calculation.general.decomposition.Chol.CholMatrix;
import org.ujmp.core.doublematrix.calculation.general.decomposition.LU.LUMatrix;
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

		// Check Argument
		if (A.isSparse()) {
			// assign default if method is not set
			if (method == null) {
				method = ProjectionMethod.AUGMENTED_SYSTEM;
			}

			// check that method is applicable for sparse A
			if (method != ProjectionMethod.AUGMENTED_SYSTEM && method != ProjectionMethod.NORMAL_EQUATION) {
				throw new RuntimeException("Method not allowed for sparse matrix.");
			}
		} else {
			// assign default if method is not set
			if (method == null) {
				method = ProjectionMethod.QR_FACTORIZATION;
			}

			// check that method is applicable for dense A
			if (method != ProjectionMethod.QR_FACTORIZATION && method != ProjectionMethod.SVD_FACTORIZATION) {
				throw new RuntimeException("Method not allowed for dense matrix.");
			}
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

		// Form augmented system:
		// [ 1 A^T ]
		// [ A  0  ]
		Matrix firstRowK = SparseMatrix2D.Factory.horCat(SparseMatrix2D.Factory.eye(A.getColumnCount(), A.getColumnCount()), A.transpose());
		Matrix secondRowK = SparseMatrix2D.Factory.horCat(A, SparseMatrix2D.Factory.zeros(A.getRowCount(), A.getRowCount()));
		Matrix K = SparseMatrix2D.Factory.vertCat(firstRowK, secondRowK);

		// LU factorization
		// TODO: Use a symmetric indefinite factorization
		//       to solve the system twice as fast (because of the symmetry).
		LUMatrix luK = new LUMatrix(K);
		if (!luK.isNonsingular()) {
			System.out.println("Singular Jacobian matrix. Using dense SVD decomposition to \n" +
					           "perform the factorizations.");
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
			public Matrix apply(Matrix x) {
				// v = [x]
			    //     [0]
				Matrix v = Matrix.Factory.zeros(x.getRowCount() + A.getRowCount(), 1);
				for (long[] pos: x.allCoordinates()) {
					v.setAsDouble(x.getAsDouble(pos), pos);
				}

				// lu_sol = [ z ]
		        //          [aux]
				Matrix luSol = luK.solve(v);
				Matrix z = luSol.subMatrix(Ret.LINK, 0, 0, x.getRowCount()-1, 0);

				// Iterative refinement to improve roundoff
				// errors described in [2]_, algorithm 5.2.
				int k = 0;
				while (orthogonality(A, z) > orthTol) {
					if (k >= maxRefine) {
						break;
					}

					// new_v = [x] - [I A.T] * [ z ]
		            //         [0]   [A  O ]   [aux]
					Matrix newV = v.minus(K.mtimes(luSol));

					// [I A.T] * [delta  z ] = new_v
		            // [A  O ]   [delta aux]
					Matrix luUpdate = luK.solve(newV);

					// [ z ] += [delta  z ]
		            // [aux]    [delta aux]
					luSol = luSol.plus(luUpdate);
					z = luSol.subMatrix(Ret.LINK, 0, 0, x.getRowCount()-1, 0);
					// TODO: need to re-link each time?
					// --> in-place addition possible?

					k++;
				}

				return z;
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
			public Matrix apply(Matrix x) {
				// v = [x]
			    //     [0]
				Matrix v = Matrix.Factory.zeros(x.getRowCount() + A.getRowCount(), 1);
				for (long[] pos: x.allCoordinates()) {
					v.setAsDouble(x.getAsDouble(pos), pos);
				}

				// lu_sol = [aux]
		        //          [ z ]
				Matrix luSol = luK.solve(v);

				// return z = inv(A A.T) A x
				return luSol.subMatrix(Ret.NEW, x.getRowCount(), 0, luSol.getRowCount()-1, 0);
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
			public Matrix apply(Matrix x) {
				// v = [0]
		        //     [x]
				long rowsA = A.getColumnCount();
				Matrix v = Matrix.Factory.zeros(rowsA + x.getRowCount(), 1);
				for (long[] pos: x.allCoordinates()) {
					v.setAsDouble(x.getAsDouble(pos), rowsA + pos[0], pos[1]);
				}

				// lu_sol = [ z ]
		        //          [aux]
				Matrix luSol = luK.solve(v);

				// return z = A.T inv(A A.T) x
				Matrix z = luSol.subMatrix(Ret.NEW, 0, 0, rowsA-1, 0);
				return z;
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
