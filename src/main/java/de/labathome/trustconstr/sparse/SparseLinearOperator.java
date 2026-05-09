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
package de.labathome.trustconstr.sparse;

import de.labathome.trustconstr.matrix.MatrixOps;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Adapter from a {@link CSRMatrix} or {@link CSCMatrix} to the existing
 * {@link LinearOperator} interface used throughout the algorithm. Mirrors
 * the role of {@code scipy.sparse.linalg.LinearOperator} produced from a
 * sparse matrix via {@code aslinearoperator(A)}.
 *
 * <p>The {@link LinearOperator} contract is typed in terms of {@link Matrix},
 * so this adapter does the {@code double[]} &harr; {@link Matrix} conversion
 * at the boundary on each apply.
 */
public final class SparseLinearOperator {

	private SparseLinearOperator() { }

	/**
	 * @param a CSR matrix
	 * @return a {@link LinearOperator} that computes {@code y = A x}
	 */
	public static LinearOperator forMatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSC matrix
	 * @return a {@link LinearOperator} that computes {@code y = A x}
	 */
	public static LinearOperator forMatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSR matrix
	 * @return a {@link LinearOperator} that computes {@code y = A^T x}
	 */
	public static LinearOperator forRmatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSC matrix
	 * @return a {@link LinearOperator} that computes {@code y = A^T x}
	 */
	public static LinearOperator forRmatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}
}
