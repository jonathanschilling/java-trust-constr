package de.labathome.optimization;

import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DoubleMatrix2D;

public class Projections {

	/**
	 * Measure orthogonality between a vector and the null space of a matrix.
	 *
	 * Compute a measure of orthogonality between the null space
	 * of the (possibly sparse) matrix ``A`` and a given vector ``g``.
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

	public static Matrix[] projections(DoubleMatrix2D A)  {
		ProjectionMethod method = null;
		return projections(A, method);
	}

	public static Matrix[] projections(DoubleMatrix2D A, ProjectionMethod method)  {
		double orthTol = 1.0e-12;
		return projections(A, method, orthTol);
	}

	public static Matrix[] projections(DoubleMatrix2D A, ProjectionMethod method, double orthTol)  {
		int maxRefine = 3;
		return projections(A, method, orthTol, maxRefine);
	}

	public static Matrix[] projections(DoubleMatrix2D A, ProjectionMethod method, double orthTol, int maxRefine)  {
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
	public static Matrix[] projections(DoubleMatrix2D A, ProjectionMethod method, double orthTol, int maxRefine, double tolerance)  {

		// Check Argument
		if (A.isSparse()) {
			if (method == null) {
				method = ProjectionMethod.AUGMENTED_SYSTEM;
			}

			if (method != ProjectionMethod.AUGMENTED_SYSTEM && method != ProjectionMethod.NORMAL_EQUATION) {
				throw new RuntimeException("Method not allowed for sparse matrix.");
			}
		} else {
			if (method == null) {
				method = ProjectionMethod.QR_FACTORIZATION;
			} else if (method != ProjectionMethod.QR_FACTORIZATION && method != ProjectionMethod.SVD_FACTORIZATION) {
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
	 * Return linear operators for matrix A using ``NormalEquation`` approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return
	 */
	private static Matrix[] normalEquationProjections(DoubleMatrix2D A, double orthTol, int maxRefine, double tolerance) {

		return null;
	}

	/**
	 * Return linear operators for matrix A - ``AugmentedSystem``.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return
	 */
	private static Matrix[] augmentedSystemProjections(DoubleMatrix2D A, double orthTol, int maxRefine, double tolerance) {

		return null;
	}

	/**
	 * Return linear operators for matrix A using ``QRFactorization`` approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return
	 */
	private static Matrix[] qrFactorizationProjections(DoubleMatrix2D A, double orthTol, int maxRefine, double tolerance) {

		return null;
	}

	/**
	 * Return linear operators for matrix A using ``SVDFactorization`` approach.
	 *
	 * @param A
	 * @param m
	 * @param n
	 * @param orthTol
	 * @param maxRefine
	 * @param tolerance
	 * @return
	 */
	private static Matrix[] svdFactorizationProjections(DoubleMatrix2D A, double orthTol, int maxRefine, double tolerance) {

		return null;
	}
}
