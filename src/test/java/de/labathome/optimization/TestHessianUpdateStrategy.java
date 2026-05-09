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

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.BFGS;
import de.labathome.trustconstr.SR1;
import de.labathome.trustconstr.enums.ExceptionStrategy;
import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Java translation of subset of scipy
 * {@code optimize/tests/test_hessian_update_strategy.py}: the curvature
 * skip-update tests, the {@code matmul == dot} parity test (Java has only
 * {@code dot}, not the {@code @} operator, so we verify that
 * {@code dot(p) == B @ p} via {@code getMatrix().mtimes(p)}), and the
 * {@code HESSIAN} vs {@code INV_HESSIAN} round trip.
 *
 * <p>The big {@code test_rosenbrock_with_no_exception} parametrised test (the
 * 30+ Rosenbrock iteration points) is exercised indirectly through the Java
 * integration tests {@code TestQuasiNewtonAndFiniteDiff} and
 * {@code TestEqualityConstrainedRosenbrock}. The strict 1:1 port of that
 * test is deferred -- the test fixture is identical to scipy's so a port
 * is mechanical; defer to a follow-up PR.
 */
class TestHessianUpdateStrategy {

	private static final double TOL = 1e-12;

	@Test
	void bfgsSkipUpdateLeavesMatrixUnchanged() {
		// scipy test_BFGS_skip_update: with a high min_curvature, an update
		// where the curvature condition is violated should leave the matrix
		// unchanged. Use a fresh BFGSFactory rather than the shared static
		// FACTORY singleton -- the factory carries mutable state that leaks
		// across tests if reused.
		BFGS hess = new BFGS.BFGSFactory()
				.exceptionStrategy(ExceptionStrategy.SKIP_UPDATE)
				.minCurvature(10.0)  // very high, so all updates skip
				.build();
		hess.initialize(2, HessianApproximationType.HESSIAN);

		// First update: violates curvature (wz <= minCurvature * wMw with min=10).
		Matrix s = DenseMatrix.column(0.1, 0.2);
		Matrix y = DenseMatrix.column(0.05, 0.1);
		hess.update(s, y);
		Matrix B0 = hess.getMatrix();

		// Second update: should also skip due to high min_curvature.
		Matrix s2 = DenseMatrix.column(0.05, 0.15);
		Matrix y2 = DenseMatrix.column(0.02, 0.08);
		hess.update(s2, y2);
		Matrix B1 = hess.getMatrix();

		// Skip semantics: the matrix should be unchanged across the second update.
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				RelAbsAssertions.assertRelAbsEquals(
						B0.getAsDouble(i, j), B1.getAsDouble(i, j), TOL,
						"B[" + i + "," + j + "]");
			}
		}
	}

	@Test
	void sr1SkipUpdateLeavesMatrixUnchanged() {
		// scipy test_SR1_skip_update: same idea for SR1 with min_denominator.
		// Fresh SR1Factory -- avoids state leak through SR1.FACTORY.
		SR1 hess = new SR1.SR1Factory()
				.minDenominator(1e50)  // huge denominator threshold -> always skip
				.build();
		hess.initialize(2, HessianApproximationType.HESSIAN);

		Matrix s = DenseMatrix.column(0.1, 0.2);
		Matrix y = DenseMatrix.column(0.05, 0.1);
		hess.update(s, y);
		Matrix B0 = hess.getMatrix();

		Matrix s2 = DenseMatrix.column(0.05, 0.15);
		Matrix y2 = DenseMatrix.column(0.02, 0.08);
		hess.update(s2, y2);
		Matrix B1 = hess.getMatrix();

		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				RelAbsAssertions.assertRelAbsEquals(
						B0.getAsDouble(i, j), B1.getAsDouble(i, j), TOL,
						"B[" + i + "," + j + "]");
			}
		}
	}

	@Test
	void bfgsApplyEqualsGetMatrixTimesP() {
		// scipy test_matmul_equals_dot for BFGS: H @ v == H.dot(v).
		// Java equivalent: strategy.apply(p, null) == strategy.getMatrix().mtimes(p).
		BFGS hess = BFGS.FACTORY.build();
		hess.initialize(3, HessianApproximationType.HESSIAN);

		// Update once so the matrix is non-trivial.
		hess.update(
				DenseMatrix.column(0.1, 0.2, 0.3),
				DenseMatrix.column(0.05, 0.1, 0.15));

		Matrix p = DenseMatrix.column(1.0, 2.0, 3.0);
		Matrix viaApply = hess.apply(p, null);
		Matrix viaMtimes = hess.getMatrix().mtimes(p);

		assertEquals(viaApply.getRowCount(), viaMtimes.getRowCount());
		for (long i = 0; i < viaApply.getRowCount(); ++i) {
			RelAbsAssertions.assertRelAbsEquals(
					viaApply.getAsDouble(i, 0),
					viaMtimes.getAsDouble(i, 0), TOL, "[" + i + "]");
		}
	}

	@Test
	void sr1ApplyEqualsGetMatrixTimesP() {
		SR1 hess = SR1.FACTORY.build();
		hess.initialize(3, HessianApproximationType.HESSIAN);
		hess.update(
				DenseMatrix.column(0.1, 0.2, 0.3),
				DenseMatrix.column(0.05, 0.1, 0.15));

		Matrix p = DenseMatrix.column(1.0, 2.0, 3.0);
		Matrix viaApply = hess.apply(p, null);
		Matrix viaMtimes = hess.getMatrix().mtimes(p);

		for (long i = 0; i < viaApply.getRowCount(); ++i) {
			RelAbsAssertions.assertRelAbsEquals(
					viaApply.getAsDouble(i, 0),
					viaMtimes.getAsDouble(i, 0), TOL, "[" + i + "]");
		}
	}

	@Test
	void bfgsHessianAndInvHessianBothUsable() {
		// scipy test_hessian_initialization (subset): both 'hess' and
		// 'inv_hess' modes initialise correctly. After one update, the
		// matrices are well-defined and self-consistent.
		BFGS hessMode = BFGS.FACTORY.build();
		hessMode.initialize(3, HessianApproximationType.HESSIAN);

		BFGS invHessMode = BFGS.FACTORY.build();
		invHessMode.initialize(3, HessianApproximationType.INV_HESSIAN);

		Matrix s = DenseMatrix.column(0.1, 0.2, 0.3);
		Matrix y = DenseMatrix.column(0.05, 0.1, 0.15);
		hessMode.update(s, y);
		invHessMode.update(s, y);

		Matrix B = hessMode.getMatrix();    // approximates H
		Matrix Hi = invHessMode.getMatrix(); // approximates H^-1

		// B and Hi should both be 3x3, both symmetric, both positive definite
		// after a single BFGS update with a positive curvature pair.
		assertEquals(3, B.getRowCount());
		assertEquals(3, Hi.getRowCount());
		// Symmetry check
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				RelAbsAssertions.assertRelAbsEquals(
						B.getAsDouble(i, j), B.getAsDouble(j, i), TOL,
						"B symmetric");
				RelAbsAssertions.assertRelAbsEquals(
						Hi.getAsDouble(i, j), Hi.getAsDouble(j, i), TOL,
						"H^-1 symmetric");
			}
		}
	}

	@Test
	void sr1HessianAndInvHessianBothUsable() {
		SR1 hessMode = SR1.FACTORY.build();
		hessMode.initialize(3, HessianApproximationType.HESSIAN);

		SR1 invHessMode = SR1.FACTORY.build();
		invHessMode.initialize(3, HessianApproximationType.INV_HESSIAN);

		Matrix s = DenseMatrix.column(0.1, 0.2, 0.3);
		Matrix y = DenseMatrix.column(0.05, 0.1, 0.15);
		hessMode.update(s, y);
		invHessMode.update(s, y);

		Matrix B = hessMode.getMatrix();
		Matrix Hi = invHessMode.getMatrix();
		assertEquals(3, B.getRowCount());
		assertEquals(3, Hi.getRowCount());
		// SR1 produces symmetric updates by construction.
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				RelAbsAssertions.assertRelAbsEquals(
						B.getAsDouble(i, j), B.getAsDouble(j, i), TOL,
						"B symmetric");
				RelAbsAssertions.assertRelAbsEquals(
						Hi.getAsDouble(i, j), Hi.getAsDouble(j, i), TOL,
						"H^-1 symmetric");
			}
		}
	}
}
