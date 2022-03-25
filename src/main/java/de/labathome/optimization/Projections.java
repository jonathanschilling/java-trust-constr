package de.labathome.optimization;

import de.labathome.LinAlg;

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
	public static double orthogonality(double[][] A, double[] g) {

		double normG = LinAlg.norm(g);

		// TODO: sparse version of A
		double normA = LinAlg.frob(A);

		// Check if norms are zero
		if (normG == 0.0 || normA == 0.0) {
			return 0.0;
		}

		double normAg = LinAlg.norm(LinAlg.dot(A, g));

		// Orthogonality measure
		double orth = normAg/(normA * normG);

		return orth;
	}

}
