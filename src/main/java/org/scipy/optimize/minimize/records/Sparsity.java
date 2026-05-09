package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.NumDiff;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Pairing of a sparsity-pattern matrix {@code A} with a precomputed
 * Curtis-Powell-Reid column grouping. Used by
 * {@link org.scipy.optimize.minimize.NumDiff#approxDerivative} to drive the
 * sparse-Jacobian finite-difference path.
 */
public final class Sparsity {

	private Matrix A;
	private int[] sparsityGroups;

	/**
	 * @param A              {@code m x n} sparsity-pattern matrix (non-zero
	 *                       entries mark structural non-zeros of the Jacobian)
	 * @param sparsityGroups length-{@code n} column-group assignment from
	 *                       {@link NumDiff#groupColumns(Matrix)}
	 */
	public Sparsity(Matrix A, int[] sparsityGroups) {
		this.A = A;
		this.sparsityGroups = sparsityGroups;
	}

	/** @return the sparsity-pattern matrix {@code A} */
	public Matrix A() {
		return A;
	}

	/** @return per-column group indices */
	public int[] sparsityGroups() {
		return sparsityGroups;
	}

	/**
	 * Convenience factory: derive the column grouping from {@code A} via
	 * {@link NumDiff#groupColumns(Matrix)}.
	 *
	 * @param A sparsity-pattern matrix
	 * @return a fresh {@link Sparsity}
	 */
	public static Sparsity of(Matrix A) {
		int[] sparsityGroups = NumDiff.groupColumns(A);
		return new Sparsity(A, sparsityGroups);
	}

	/**
	 * Convenience factory with a user-supplied column ordering.
	 *
	 * @param A     sparsity-pattern matrix
	 * @param order permutation array passed through to
	 *              {@link NumDiff#groupColumns(Matrix, int[])}
	 * @return a fresh {@link Sparsity}
	 */
	public static Sparsity of(Matrix A, int[] order) {
		int[] sparsityGroups = NumDiff.groupColumns(A, order);
		return new Sparsity(A, sparsityGroups);
	}
}
