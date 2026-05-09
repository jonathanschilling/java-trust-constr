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
package de.labathome.optimization;

import static de.labathome.optimization.RelAbsAssertions.assertArrayRelAbsEquals;

import java.util.function.Function;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.NumDiff;
import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.FiniteDifferenceBounds;
import de.labathome.trustconstr.records.FiniteDifferenceOptions;
import de.labathome.trustconstr.records.Sparsity;

/**
 * Exercises {@link NumDiff#approxDerivative} on the sparse code path.
 * Until this test landed, {@code sparseDifference} was unreachable from any
 * test fixture (no caller in src/ ever sets {@code finiteDiffJacSparsity} on
 * {@code VectorFunction}'s factory), which let a latent indexing bug -- using
 * a Jacobian-row index where a Jacobian-column index was meant -- go silent
 * for any rectangular Jacobian.
 *
 * <p>The cases below cover {@code m != n} in both directions plus {@code m == n}
 * as a control.
 */
class TestSparseDifference {

	private static final double TOL = 1e-6;

	/**
	 * f : R^4 -> R^3, mixed linear/quadratic. Sparsity:
	 * {@code [[1,0,1,0],[0,1,0,1],[1,0,0,1]]}.
	 * Curtis-Powell-Reid groups: {col0,col1} and {col2,col3}.
	 */
	@Test
	void rectangularMlessThanN() {
		Function<Matrix, Matrix> fun = x -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(2, 0),
				3.0 * x.getAsDouble(1, 0) + x.getAsDouble(3, 0) * x.getAsDouble(3, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(3, 0));

		Matrix x0 = DenseMatrix.column(0.7, -0.3, 1.4, 2.1);

		double[][] sparsityPattern = {
				{1, 0, 1, 0},
				{0, 1, 0, 1},
				{1, 0, 0, 1},
		};
		Matrix structure = DenseMatrix.fromRows(sparsityPattern);
		Sparsity sparsity = Sparsity.of(structure);

		// Analytic Jacobian, row-major as the [[]] layout above.
		double[][] expected = {
				{2.0 * 0.7,  0.0,         1.0,  0.0},
				{0.0,        3.0,         0.0,  2.0 * 2.1},
				{1.0,        0.0,         0.0,  1.0},
		};

		FiniteDifferenceOptions options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.TWO_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(x0.getRowCount()))
				.sparsity(sparsity)
				.build();

		Matrix J = NumDiff.approxDerivative(fun, x0, null, options);

		assertArrayRelAbsEquals(expected, J.toDoubleArray(), TOL);
	}

	/**
	 * f : R^2 -> R^4, m > n: this is the case that would have crashed the old
	 * sparse path with an ArrayIndexOutOfBoundsException (the old code indexed
	 * a length-n group-mask {@code e} with a row index in {@code [0, m)}; for
	 * m > n the access reads past the array).
	 */
	@Test
	void rectangularMgreaterThanN() {
		Function<Matrix, Matrix> fun = x -> DenseMatrix.column(
				2.0 * x.getAsDouble(0, 0),
				3.0 * x.getAsDouble(1, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(1, 0),
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0));

		Matrix x0 = DenseMatrix.column(0.5, -1.2);

		double[][] sparsityPattern = {
				{1, 0},
				{0, 1},
				{1, 1},
				{1, 0},
		};
		Matrix structure = DenseMatrix.fromRows(sparsityPattern);
		Sparsity sparsity = Sparsity.of(structure);

		double[][] expected = {
				{2.0,        0.0},
				{0.0,        3.0},
				{1.0,        1.0},
				{2.0 * 0.5,  0.0},
		};

		FiniteDifferenceOptions options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.TWO_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(x0.getRowCount()))
				.sparsity(sparsity)
				.build();

		Matrix J = NumDiff.approxDerivative(fun, x0, null, options);

		assertArrayRelAbsEquals(expected, J.toDoubleArray(), TOL);
	}

	/**
	 * f : R^4 -> R^4, m == n control. The old buggy code happened to work in
	 * this case (the e[row] read is a valid index when m == n), so a parity
	 * test that only ever ran m == n would have missed the latent bug.
	 */
	@Test
	void squareMequalsN() {
		Function<Matrix, Matrix> fun = x -> DenseMatrix.column(
				2.0 * x.getAsDouble(0, 0),
				3.0 * x.getAsDouble(1, 0),
				x.getAsDouble(2, 0) * x.getAsDouble(2, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(3, 0));

		Matrix x0 = DenseMatrix.column(0.7, -0.4, 1.1, 2.0);

		double[][] sparsityPattern = {
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{1, 0, 0, 1},
		};
		Matrix structure = DenseMatrix.fromRows(sparsityPattern);
		Sparsity sparsity = Sparsity.of(structure);

		double[][] expected = {
				{2.0,  0.0,         0.0,         0.0},
				{0.0,  3.0,         0.0,         0.0},
				{0.0,  0.0,         2.0 * 1.1,   0.0},
				{1.0,  0.0,         0.0,         1.0},
		};

		FiniteDifferenceOptions options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.TWO_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(x0.getRowCount()))
				.sparsity(sparsity)
				.build();

		Matrix J = NumDiff.approxDerivative(fun, x0, null, options);

		assertArrayRelAbsEquals(expected, J.toDoubleArray(), TOL);
	}

	/**
	 * Same rectangular m &lt; n setup but with the THREE_POINT (central-difference)
	 * scheme. Exercises the second branch in {@code sparseDifference}.
	 */
	@Test
	void rectangularThreePoint() {
		Function<Matrix, Matrix> fun = x -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(2, 0),
				3.0 * x.getAsDouble(1, 0) + x.getAsDouble(3, 0) * x.getAsDouble(3, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(3, 0));

		Matrix x0 = DenseMatrix.column(0.7, -0.3, 1.4, 2.1);

		double[][] sparsityPattern = {
				{1, 0, 1, 0},
				{0, 1, 0, 1},
				{1, 0, 0, 1},
		};
		Matrix structure = DenseMatrix.fromRows(sparsityPattern);
		Sparsity sparsity = Sparsity.of(structure);

		double[][] expected = {
				{2.0 * 0.7,  0.0,  1.0,  0.0},
				{0.0,        3.0,  0.0,  2.0 * 2.1},
				{1.0,        0.0,  0.0,  1.0},
		};

		FiniteDifferenceOptions options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.THREE_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(x0.getRowCount()))
				.sparsity(sparsity)
				.build();

		Matrix J = NumDiff.approxDerivative(fun, x0, null, options);

		assertArrayRelAbsEquals(expected, J.toDoubleArray(), TOL);
	}
}
