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

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.NonlinearVectorFunction;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Java translation of scipy
 * {@code optimize/tests/test_differentiable_functions.py}: subset of the
 * {@code TestScalarFunction} class tests, plus a smoke test for the
 * Java-port-specific {@link NonlinearVectorFunction} (the analytic-derivatives
 * vector-function carrier introduced in Phase 2).
 *
 * <p>The full-shape {@code TestScalarFunction.test_finite_difference_grad},
 * {@code test_finite_difference_hess_linear_operator}, and
 * {@code test_x_storage_overlap} tests target scipy's heavier
 * {@code ScalarFunction} class with its FD/BFGS fallback machinery; those
 * code paths are exercised indirectly through the Java port's
 * trust-constr integration tests, and a strict 1:1 port of the remaining
 * scipy tests is deferred.
 */
class TestScalarFunction {

	private static final double TOL = 1e-12;

	@Test
	void nonlinearVectorFunctionCachesFAndJ() {
		// f(x) = (x[0]^2 + x[1]^2, x[0] - x[1]) -- 2D nonlinear vector fn.
		Function<Matrix, Matrix> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return DenseMatrix.column(a * a + b * b, a - b);
		};
		Function<Matrix, Matrix> jac = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return DenseMatrix.fromRows(new double[][] {{2 * a, 2 * b}, {1, -1}});
		};
		BiFunction<Matrix, Matrix, Matrix> hess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			return DenseMatrix.fromRows(new double[][] {{2 * v0, 0}, {0, 2 * v0}});
		};

		Matrix x0 = DenseMatrix.column(1.0, 0.5);
		NonlinearVectorFunction nvf = new NonlinearVectorFunction(
				fun, jac, hess, x0, Optional.empty());

		// Cached f = (1.25, 0.5) at x0
		Matrix f = nvf.f();
		RelAbsAssertions.assertRelAbsEquals(1.25, f.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(1, 0), TOL);

		// Cached J at x0 = ((2, 1), (1, -1))
		Matrix J = nvf.J();
		RelAbsAssertions.assertRelAbsEquals(2.0, J.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.0, J.getAsDouble(0, 1), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.0, J.getAsDouble(1, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(-1.0, J.getAsDouble(1, 1), TOL);

		// Initial eval counts: 1 each
		assertEquals(1, nvf.numFunctionEvals(), "constructor evaluates f once");
		assertEquals(1, nvf.numJacobianEvals(), "constructor evaluates J once");
		assertEquals(0, nvf.numHessianEvals(), "no Hessian eval yet");
	}

	@Test
	void nonlinearVectorFunctionUpdatesOnXChange() {
		// Verify that fun(x) and jac(x) update the cache when x changes.
		Function<Matrix, Matrix> fun = x -> {
			double a = x.getAsDouble(0, 0);
			return DenseMatrix.column(a * a);
		};
		Function<Matrix, Matrix> jac = x -> {
			double a = x.getAsDouble(0, 0);
			return DenseMatrix.fromRows(new double[][] {{2 * a}});
		};
		Matrix x0 = DenseMatrix.column(1.0);
		NonlinearVectorFunction nvf = new NonlinearVectorFunction(
				fun, jac, null, x0, Optional.empty());

		// Same x -> no re-eval
		Matrix f0Again = nvf.fun(x0);
		assertEquals(1, nvf.numFunctionEvals(), "same x reuses cache");
		RelAbsAssertions.assertRelAbsEquals(1.0, f0Again.getAsDouble(0, 0), TOL);

		// Different x -> re-eval
		Matrix x1 = DenseMatrix.column(2.0);
		Matrix f1 = nvf.fun(x1);
		RelAbsAssertions.assertRelAbsEquals(4.0, f1.getAsDouble(0, 0), TOL);
		assertEquals(2, nvf.numFunctionEvals(), "x changed -> +1 eval");

		// jac at the new x
		Matrix J1 = nvf.jac(x1);
		RelAbsAssertions.assertRelAbsEquals(4.0, J1.getAsDouble(0, 0), TOL);
		assertEquals(2, nvf.numJacobianEvals(), "x changed -> +1 jac eval");
	}

	@Test
	void nonlinearVectorFunctionHessReturnsZerosWhenNoUserHess() {
		// When userHess is null, hess(x, v) returns the n x n zero matrix
		// (rather than throwing or returning null). Mirrors scipy's behaviour
		// when hess is omitted -- in scipy, BFGS is used as fallback; in our
		// Phase-2 Java port, NonlinearVectorFunction simply returns zeros and
		// the orchestrator dispatches to a different Hessian source.
		Function<Matrix, Matrix> fun = x -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0));
		Function<Matrix, Matrix> jac = x -> DenseMatrix.fromRows(new double[][] {
				{2 * x.getAsDouble(0, 0)}});
		NonlinearVectorFunction nvf = new NonlinearVectorFunction(
				fun, jac, null, DenseMatrix.column(1.0), Optional.empty());

		Matrix v = DenseMatrix.column(0.5);
		Matrix H = nvf.hess(DenseMatrix.column(1.0), v);
		assertEquals(1, H.getRowCount());
		assertEquals(1, H.getColumnCount());
		assertEquals(0.0, H.getAsDouble(0, 0));
	}
}
