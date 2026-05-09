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
package de.labathome.trustconstr.matrix;

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Cholesky factorisation of a symmetric positive-definite matrix.
 * {@code factor} holds the LAPACK {@code dpotrf} output verbatim -- the upper
 * (if {@code upper}) or lower triangle stores the Cholesky factor; the other
 * triangle is unspecified. Reuse the captured factor for repeated solves
 * via {@link #solve(DenseMatrix)} (one {@code dpotrs} call, no re-factoring).
 */
public record CholResult(double[] factor, int n, boolean upper) {

	/**
	 * Solve {@code A x = rhs} where {@code A} is the original SPD matrix
	 * factored into {@link #factor()}. Multiple right-hand sides supported
	 * via the column count of {@code rhs}. Does not modify {@link #factor()}.
	 *
	 * @param rhs right-hand side ({@code n x nrhs}); must have the same row
	 *            count as the factored matrix
	 * @return the solution {@code x} ({@code n x nrhs})
	 */
	public DenseMatrix solve(DenseMatrix rhs) {
		if (rhs.rows() != n) {
			throw new IllegalArgumentException("CholResult.solve: rhs row count "
					+ rhs.rows() + " != factor dimension " + n);
		}
		int nrhs = rhs.cols();
		double[] bFlat = rhs.data().clone();
		intW info = new intW(0);
		LAPACK.getInstance().dpotrs(upper ? "U" : "L", n, nrhs, factor, n, bFlat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dpotrs failed: info=" + info.val);
		}
		return DenseMatrix.fromColumnMajor(n, nrhs, bFlat);
	}
}
