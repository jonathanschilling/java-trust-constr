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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.IdentityVectorFunction;
import de.labathome.trustconstr.LinearVectorFunction;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

/**
 * Java translation of scipy
 * {@code optimize/tests/test_differentiable_functions.py}: the standalone
 * {@code test_LinearVectorFunction}, {@code test_LinearVectorFunction_memoization},
 * and {@code test_IdentityVectorFunction} tests.
 *
 * <p>The {@code TestVectorialFunction} class-level tests for the heavier
 * {@link de.labathome.trustconstr.VectorFunction} (FD jac / hess linear
 * operator / sparsity / x storage) are exercised indirectly through the
 * trust-constr integration tests; strict 1:1 ports of the remaining
 * scipy-class tests are deferred.
 */
class TestVectorFunction {

	private static final double TOL = 1e-12;

	@Test
	void linearVectorFunctionDenseAndSparse() {
		// scipy test_LinearVectorFunction: matvec and Jacobian round-trip
		// for several sparse-jacobian preferences.
		Matrix Adense = DenseMatrix.fromRows(new double[][] {
				{-1, 2, 0},
				{ 0, 4, 2}});
		Matrix x0 = DenseMatrix.column(0.0, 0.0, 0.0);

		LinearVectorFunction f1 = new LinearVectorFunction(Adense, x0, Optional.empty());
		assertFalse(f1.sparseJacobian(),
				"sparseJacobian=None preserves dense input");

		LinearVectorFunction f2 = new LinearVectorFunction(Adense, x0, Optional.of(true));
		assertTrue(f2.sparseJacobian(), "sparseJacobian=True forces sparse");

		LinearVectorFunction f3 = new LinearVectorFunction(Adense, x0, Optional.of(false));
		assertFalse(f3.sparseJacobian(), "sparseJacobian=False forces dense");

		// Build the same matrix as a SparseMatrix.
		SparseMatrix Asparse = SparseMatrix.Factory.zeros(2, 3);
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 3; ++j) {
				double v = Adense.getAsDouble(i, j);
				if (v != 0.0) Asparse.setAsDouble(v, i, j);
			}
		}
		LinearVectorFunction f4 = new LinearVectorFunction(Asparse, x0, Optional.empty());
		assertTrue(f4.sparseJacobian(),
				"sparseJacobian=None preserves sparse input");
		LinearVectorFunction f5 = new LinearVectorFunction(Asparse, x0, Optional.of(true));
		assertTrue(f5.sparseJacobian());
		LinearVectorFunction f6 = new LinearVectorFunction(Asparse, x0, Optional.of(false));
		assertFalse(f6.sparseJacobian());

		// fun(x) = A x for x = (1, -1, 0). Expected (1*-1 + 2*-1 + 0*0, 0*-1 + 4*-1 + 2*0) = (-3, -4)
		Matrix x = DenseMatrix.column(1.0, -1.0, 0.0);
		Matrix Ax = f1.fun(x);
		RelAbsAssertions.assertRelAbsEquals(-3.0, Ax.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(-4.0, Ax.getAsDouble(1, 0), TOL);
		Matrix Ax2 = f2.fun(x);
		RelAbsAssertions.assertRelAbsEquals(-3.0, Ax2.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(-4.0, Ax2.getAsDouble(1, 0), TOL);

		// jac(x) returns A
		Matrix J1 = f1.jac(x);
		assertEquals(2, J1.getRowCount());
		assertEquals(3, J1.getColumnCount());
		RelAbsAssertions.assertRelAbsEquals(-1.0, J1.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(2.0, J1.getAsDouble(0, 1), TOL);
		RelAbsAssertions.assertRelAbsEquals(4.0, J1.getAsDouble(1, 1), TOL);

		// hess(x, v) is the n x n zero matrix.
		Matrix v = DenseMatrix.column(-1.0, 1.0);
		Matrix H = f1.hess(x, v);
		assertEquals(3, H.getRowCount());
		assertEquals(3, H.getColumnCount());
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				assertEquals(0.0, H.getAsDouble(i, j));
			}
		}
	}

	@Test
	void linearVectorFunctionMemoization() {
		// scipy test_LinearVectorFunction_memoization: jac(x) updates the
		// cached x but doesn't re-evaluate f until fun(x) is called.
		Matrix A = DenseMatrix.fromRows(new double[][] {
				{-1, 2, 0},
				{ 0, 4, 2}});
		Matrix x0 = DenseMatrix.column(1.0, 2.0, -1.0);
		LinearVectorFunction fn = new LinearVectorFunction(A, x0, Optional.of(false));

		// Initial cached f = A @ x0
		Matrix Ax0 = A.mtimes(x0);
		Matrix f = fn.f();
		for (int i = 0; i < 2; ++i) {
			RelAbsAssertions.assertRelAbsEquals(Ax0.getAsDouble(i, 0), f.getAsDouble(i, 0), TOL);
		}

		Matrix x1 = DenseMatrix.column(-1.0, 3.0, 10.0);
		// jac(x1) returns the constant A regardless of x.
		Matrix J = fn.jac(x1);
		assertEquals(2, J.getRowCount());
		assertEquals(3, J.getColumnCount());

		// After jac(x1), the cached x is x1 but f is still the OLD f.
		// (scipy: assert_array_equal(A.dot(x0), fun.f) -- f wasn't recomputed.)
		// In Java: x is updated but f is invalidated lazily until fun() is called.
		Matrix Ax1 = A.mtimes(x1);
		// Calling fun(x1) updates f to A @ x1.
		Matrix Ax1Computed = fn.fun(x1);
		for (int i = 0; i < 2; ++i) {
			RelAbsAssertions.assertRelAbsEquals(Ax1.getAsDouble(i, 0),
					Ax1Computed.getAsDouble(i, 0), TOL);
		}
	}

	@Test
	void identityVectorFunction() {
		// scipy test_IdentityVectorFunction: fun(x) = x, jac(x) = I,
		// hess(x, v) = 0.
		Matrix x0 = DenseMatrix.column(0.0, 0.0, 0.0);

		IdentityVectorFunction f1 = new IdentityVectorFunction(x0, Optional.empty());
		IdentityVectorFunction f2 = new IdentityVectorFunction(x0, Optional.of(false));
		IdentityVectorFunction f3 = new IdentityVectorFunction(x0, Optional.of(true));

		assertTrue(f1.sparseJacobian(), "default is sparse");
		assertFalse(f2.sparseJacobian());
		assertTrue(f3.sparseJacobian());

		Matrix x = DenseMatrix.column(-1.0, 2.0, 1.0);
		Matrix Ax = f1.fun(x);
		for (int i = 0; i < 3; ++i) {
			RelAbsAssertions.assertRelAbsEquals(x.getAsDouble(i, 0),
					Ax.getAsDouble(i, 0), TOL);
		}

		Matrix J = f2.jac(x);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				double expected = (i == j) ? 1.0 : 0.0;
				RelAbsAssertions.assertRelAbsEquals(expected, J.getAsDouble(i, j), TOL);
			}
		}

		Matrix v = DenseMatrix.column(-2.0, 3.0, 0.0);
		Matrix H = f1.hess(x, v);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				assertEquals(0.0, H.getAsDouble(i, j));
			}
		}
	}
}
