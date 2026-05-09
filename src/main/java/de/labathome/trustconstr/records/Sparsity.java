/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.trustconstr.records;

import de.labathome.trustconstr.NumDiff;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Pairing of a sparsity-pattern matrix {@code A} with a precomputed
 * Curtis-Powell-Reid column grouping. Used by
 * {@link de.labathome.trustconstr.NumDiff#approxDerivative} to drive the
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
