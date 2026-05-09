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

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Thin wrapper around LAPACK direct linear solvers for the trust-constr
 * KKT path.
 *
 * <p>The trust-region constrained algorithm needs to solve the augmented
 * KKT system
 *
 * <pre>
 *     [ I    A^T ] [ x ]   [ b1 ]
 *     [ A     0  ] [ y ] = [ b2 ]
 * </pre>
 *
 * inside both the AugmentedSystem projection
 * ({@code projections.py:96-176}) and {@code eqp_kktfact}
 * ({@code qp_subproblem.py:50-58}). The Python implementation uses
 * {@code scipy.sparse.linalg.splu} on the assembled CSC matrix; here we
 * assemble in CSR/CSC, materialise to dense, and call LAPACK
 * {@code dgesv} via {@link LAPACK}. This keeps the project pure-Java and
 * unblocks the end-to-end algorithm -- replacing this with a real sparse
 * LU is a single-call-site upgrade in {@code Projections.augmentedSystem}
 * and {@code QPSubproblem.eqpKktFact}.
 */
public final class DenseSolve {

	private DenseSolve() { }

	/**
	 * Factorisation of a square matrix {@code A} ready to repeatedly
	 * {@link #solve(double[])} against right-hand sides.
	 *
	 * <p>The factor stores LU in the LAPACK column-major layout used by
	 * {@code dgesv}; calling {@code solve} costs only a forward/back
	 * substitution.
	 */
	public static final class LUFactor {
		private final int n;
		private final double[] lu;
		private final int[] ipiv;

		LUFactor(int n, double[] lu, int[] ipiv) {
			this.n = n;
			this.lu = lu;
			this.ipiv = ipiv;
		}

		/**
		 * Solve {@code A x = b} reusing the captured LU.
		 *
		 * @param b length-{@code n} right-hand side; not modified
		 * @return length-{@code n} solution {@code x}
		 */
		public double[] solve(double[] b) {
			if (b.length != n) {
				throw new IllegalArgumentException("RHS length must equal n");
			}
			double[] rhs = b.clone();
			intW info = new intW(0);
			LAPACK.getInstance().dgetrs("N", n, 1, lu, n, ipiv, rhs, n, info);
			if (info.val != 0) {
				throw new ArithmeticException("dgetrs failed with info=" + info.val);
			}
			return rhs;
		}

		/** @return the dimension of the factored matrix */
		public int n() { return n; }
	}

	/**
	 * Factor a dense {@code n x n} matrix {@code A} via LAPACK
	 * {@code dgetrf}. The input is given in row-major form and is not
	 * modified.
	 *
	 * @param a square row-major matrix to factor
	 * @return the captured LU factorisation
	 * @throws ArithmeticException if {@code A} is singular (LAPACK info &gt; 0)
	 */
	public static LUFactor factor(double[][] a) {
		int n = a.length;
		if (n == 0 || a[0].length != n) {
			throw new IllegalArgumentException("A must be square");
		}
		double[] flat = new double[n * n];
		for (int i = 0; i < n; ++i) {
			if (a[i].length != n) {
				throw new IllegalArgumentException("A must be rectangular");
			}
			for (int j = 0; j < n; ++j) {
				flat[j * n + i] = a[i][j];
			}
		}
		int[] ipiv = new int[n];
		intW info = new intW(0);
		LAPACK.getInstance().dgetrf(n, n, flat, n, ipiv, info);
		if (info.val < 0) {
			throw new IllegalArgumentException("dgetrf rejected argument " + (-info.val));
		}
		if (info.val > 0) {
			throw new ArithmeticException("dgetrf detected singular matrix at U(" + info.val + "," + info.val + ")");
		}
		return new LUFactor(n, flat, ipiv);
	}

	/**
	 * Convenience: factor and solve in one step. Equivalent to
	 * {@code factor(a).solve(b)} but reuses the factorisation only once,
	 * so prefer {@link #factor(double[][])} when you have multiple
	 * right-hand sides.
	 *
	 * @param a square row-major matrix to factor
	 * @param b length-{@code n} right-hand side
	 * @return length-{@code n} solution {@code x} of {@code A x = b}
	 */
	public static double[] solveLU(double[][] a, double[] b) {
		return factor(a).solve(b);
	}
}
