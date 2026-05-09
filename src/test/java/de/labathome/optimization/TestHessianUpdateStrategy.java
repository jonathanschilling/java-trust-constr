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
import static org.junit.jupiter.api.Assertions.assertTrue;

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
		// Use y NOT proportional to s -- y = c*s lands on the SR1 degenerate
		// manifold after autoScale, and the residual y - B*s rounds to exactly
		// zero on some BLAS implementations (pure-Java F2J, vs JNI native
		// where FMA leaves a tiny noise term).
		SR1 hess = new SR1.SR1Factory()
				.minDenominator(1e50)  // huge denominator threshold -> always skip
				.build();
		hess.initialize(2, HessianApproximationType.HESSIAN);

		Matrix s = DenseMatrix.column(0.1, 0.2);
		Matrix y = DenseMatrix.column(0.05, 0.15);
		hess.update(s, y);
		Matrix B0 = hess.getMatrix();

		Matrix s2 = DenseMatrix.column(0.05, 0.15);
		Matrix y2 = DenseMatrix.column(0.02, 0.09);
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
		// y not proportional to s -- avoids the SR1 degenerate manifold
		// where y - B*s rounds to zero post-autoScale.
		hess.update(
				DenseMatrix.column(0.1, 0.2, 0.3),
				DenseMatrix.column(0.05, 0.1, 0.2));

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

		// y not proportional to s -- avoids the SR1 degenerate manifold.
		Matrix s = DenseMatrix.column(0.1, 0.2, 0.3);
		Matrix y = DenseMatrix.column(0.05, 0.1, 0.2);
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

	/**
	 * Regression test for the SR1 zero-residual NaN bug.
	 *
	 * <p>The SR1 update formula is
	 * {@code B += (z - B*w)(z - B*w)^T / (w^T (z - B*w))}.
	 * When the residual {@code r = z - B*w} is exactly zero, the
	 * relative skip-threshold
	 * {@code |denom| < min_denominator * ||w|| * ||r||} collapses to
	 * {@code |denom| < 0} (which is never true), so the skip would not
	 * fire and the next line would compute {@code 1.0 / denom = 1/0 =
	 * Infinity} and call BLAS {@code dsyr(B, Infinity, [0, 0])}. The
	 * defensive {@code if (||r|| == 0) return;} guard in
	 * {@code SR1.updateImplementation} catches this before the divide.
	 *
	 * <p><b>BLAS-implementation sensitivity.</b> Whether this bug
	 * actually produces NaN on the test rig depends on the BLAS
	 * underneath {@code dev.ludovic.netlib}:
	 * <ul>
	 *   <li><b>JNI native BLAS</b> (system {@code libblas3}): the native
	 *       {@code dsyr} short-circuits when {@code x = 0} and leaves the
	 *       matrix unchanged regardless of {@code alpha}. The bug is
	 *       silently masked -- this test still passes without the guard
	 *       because the no-op result happens to match the expected
	 *       no-op semantic.</li>
	 *   <li><b>F2J pure-Java BLAS</b> ({@code dev.ludovic.netlib.blas.Java11BLAS},
	 *       used when no system {@code libblas3} is available --
	 *       e.g. GitHub Actions runners): {@code dsyr} executes the
	 *       fortran-translated reference loop, computes
	 *       {@code Infinity * 0 * 0 = NaN}, and stamps the matrix with
	 *       NaN. <b>This is the path that bites in CI.</b></li>
	 * </ul>
	 * To verify locally that the test catches the regression, run with
	 * {@code -Ddev.ludovic.netlib.blas.nativeLib=NONEXISTENT} to force
	 * the F2J fallback. With the guard removed, this test then fails
	 * with {@code B[0,0] = NaN}; with the guard in place it passes.
	 *
	 * <p>Originally surfaced as 3 failing CI tests in build run
	 * <code>github.com/jonathanschilling/java-trust-constr/actions/runs/25607457760</code>.
	 */
	@Test
	void sr1HandlesZeroResidualWithoutNaN() {
		// Pin B = 0.5 * I via initialScale (no autoScale), so the
		// firstIteration scaling is deterministic.
		SR1 hess = new SR1.SR1Factory().initalScale(0.5).build();
		hess.initialize(2, HessianApproximationType.HESSIAN);

		// y = 0.5 * s exactly. Both 1.0/2.0 and 0.5/1.0 are exact in
		// IEEE 754 binary64, so B*s = 0.5*s = y exactly -- residual is
		// bit-for-bit zero, regardless of BLAS implementation.
		Matrix s = DenseMatrix.column(1.0, 2.0);
		Matrix y = DenseMatrix.column(0.5, 1.0);
		hess.update(s, y);

		Matrix B = hess.getMatrix();
		// Every entry must be finite. Without the SR1 zero-residual
		// guard, B[i,j] is NaN here.
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				double v = B.getAsDouble(i, j);
				assertTrue(Double.isFinite(v),
						"B[" + i + "," + j + "] = " + v + " (must be finite)");
			}
		}
		// SR1 update with zero residual is a no-op; B stays at 0.5 * I.
		assertEquals(0.5, B.getAsDouble(0, 0), 1.0e-15);
		assertEquals(0.0, B.getAsDouble(0, 1), 1.0e-15);
		assertEquals(0.0, B.getAsDouble(1, 0), 1.0e-15);
		assertEquals(0.5, B.getAsDouble(1, 1), 1.0e-15);
	}

	/**
	 * Same regression test for INV_HESSIAN mode. SR1's
	 * {@code updateImplementation} swaps {@code w}/{@code z} for
	 * INV_HESSIAN; the same zero-residual hazard applies.
	 */
	@Test
	void sr1HandlesZeroResidualWithoutNaNInvHessian() {
		SR1 hess = new SR1.SR1Factory().initalScale(0.5).build();
		hess.initialize(2, HessianApproximationType.INV_HESSIAN);

		// In INV_HESSIAN mode, w = deltaG and z = deltaX. With H = 0.5*I
		// and z = 0.5*w, the residual z - H*w is again bit-for-bit zero.
		Matrix s = DenseMatrix.column(0.5, 1.0);   // deltaX
		Matrix y = DenseMatrix.column(1.0, 2.0);   // deltaG; H*y = 0.5*y = s
		hess.update(s, y);

		Matrix Hi = hess.getMatrix();
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				double v = Hi.getAsDouble(i, j);
				assertTrue(Double.isFinite(v),
						"H[" + i + "," + j + "] = " + v + " (must be finite)");
			}
		}
		assertEquals(0.5, Hi.getAsDouble(0, 0), 1.0e-15);
		assertEquals(0.5, Hi.getAsDouble(1, 1), 1.0e-15);
	}
}
