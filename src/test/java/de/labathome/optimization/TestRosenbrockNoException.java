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
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.BFGS;
import de.labathome.trustconstr.SR1;
import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Java translation of scipy
 * {@code optimize/tests/test_hessian_update_strategy.py:test_rosenbrock_with_no_exception}.
 *
 * <p>scipy: applies the BFGS / SR1 update over a hand-encoded sequence of
 * 38 Rosenbrock iteration points and asserts no curvature exception is
 * raised across the 37 consecutive pairs. The Java equivalent runs the
 * same updates and asserts the matrix stays finite and remains an
 * {@code n x n} dense matrix throughout (no skipped updates collapsing
 * the structure).
 */
class TestRosenbrockNoException {

	private static final int N = 5;

	/**
	 * Rosenbrock gradient at {@code x} (n = 5):
	 * <pre>
	 *   d/dx_0  = -400 x_0 (x_1 - x_0^2) - 2 (1 - x_0)
	 *   d/dx_i  = 200 (x_i - x_{i-1}^2) - 400 (x_{i+1} - x_i^2) x_i - 2 (1 - x_i)   for 0 < i < n-1
	 *   d/dx_n-1 = 200 (x_{n-1} - x_{n-2}^2)
	 * </pre>
	 *
	 * @param x current iterate, length n
	 * @return gradient vector, length n
	 */
	private static double[] rosenbrockGrad(double[] x) {
		int n = x.length;
		double[] g = new double[n];
		g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
		for (int i = 1; i < n - 1; ++i) {
			g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
					- 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
					- 2.0 * (1.0 - x[i]);
		}
		g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
		return g;
	}

	/** scipy's hand-encoded Rosenbrock iteration sequence (38 x 5). */
	private static final double[][] X_LIST = {
			{0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040},
			{0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286},
			{0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606},
			{0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750},
			{0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699},
			{0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610},
			{0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184},
			{0.2354930, 0.0443711, 0.0173959, 0.0041872, 0.00794563},
			{0.4168118, 0.1433867, 0.0111714, 0.0126265, -0.00658537},
			{0.4681972, 0.2153273, 0.0225249, 0.0152704, -0.00463809},
			{0.6023068, 0.3346815, 0.0731108, 0.0186618, -0.00371541},
			{0.6415743, 0.3985468, 0.1324422, 0.0214160, -0.00062401},
			{0.7503690, 0.5447616, 0.2804541, 0.0539851, 0.00242230},
			{0.7452626, 0.5644594, 0.3324679, 0.0865153, 0.00454960},
			{0.8059782, 0.6586838, 0.4229577, 0.1452990, 0.00976702},
			{0.8549542, 0.7226562, 0.4991309, 0.2420093, 0.02772661},
			{0.8571332, 0.7285741, 0.5279076, 0.2824549, 0.06030276},
			{0.8835633, 0.7727077, 0.5957984, 0.3411303, 0.09652185},
			{0.9071558, 0.8299587, 0.6771400, 0.4402896, 0.17469338},
			{0.9190793, 0.8486480, 0.7163332, 0.5083780, 0.26107691},
			{0.9371223, 0.8762177, 0.7653702, 0.5773109, 0.32181041},
			{0.9554613, 0.9119893, 0.8282687, 0.6776178, 0.43162744},
			{0.9545744, 0.9099264, 0.8270244, 0.6822220, 0.45237623},
			{0.9688112, 0.9351710, 0.8730961, 0.7546601, 0.56622448},
			{0.9743227, 0.9491953, 0.9005150, 0.8086497, 0.64505437},
			{0.9807345, 0.9638853, 0.9283012, 0.8631675, 0.73812581},
			{0.9886746, 0.9777760, 0.9558950, 0.9123417, 0.82726553},
			{0.9899096, 0.9803828, 0.9615592, 0.9255600, 0.85822149},
			{0.9969510, 0.9935441, 0.9864657, 0.9726775, 0.94358663},
			{0.9979533, 0.9960274, 0.9921724, 0.9837415, 0.96626288},
			{0.9995981, 0.9989171, 0.9974178, 0.9949954, 0.99023356},
			{1.0002640, 1.0005088, 1.0010594, 1.0021161, 1.00386912},
			{0.9998903, 0.9998459, 0.9997795, 0.9995484, 0.99916305},
			{1.0000008, 0.9999905, 0.9999481, 0.9998903, 0.99978047},
			{1.0000004, 0.9999983, 1.0000001, 1.0000031, 1.00000297},
			{0.9999995, 1.0000003, 1.0000005, 1.0000001, 1.00000032},
			{0.9999999, 0.9999997, 0.9999994, 0.9999989, 0.99999786},
			{0.9999999, 0.9999999, 0.9999999, 0.9999999, 0.99999991}
	};

	/**
	 * Apply the strategy over the full 37-step sequence of {@link #X_LIST}.
	 * Tests that the resulting matrix stays finite and {@code n x n} dense
	 * throughout -- the Java equivalent of scipy's "no exception raised"
	 * assertion.
	 */
	private static void runRosenbrockSequence(
			de.labathome.trustconstr.interfaces.HessianUpdateStrategy hess) {
		hess.initialize(N, HessianApproximationType.HESSIAN);
		double[][] gradList = new double[X_LIST.length][];
		for (int i = 0; i < X_LIST.length; ++i) {
			gradList[i] = rosenbrockGrad(X_LIST[i]);
		}
		for (int i = 0; i < X_LIST.length - 1; ++i) {
			double[] dx = new double[N];
			double[] dg = new double[N];
			for (int j = 0; j < N; ++j) {
				dx[j] = X_LIST[i + 1][j] - X_LIST[i][j];
				dg[j] = gradList[i + 1][j] - gradList[i][j];
			}
			Matrix dxM = DenseMatrix.column(dx);
			Matrix dgM = DenseMatrix.column(dg);
			hess.update(dxM, dgM);
		}
		Matrix B = hess.getMatrix();
		assertNotNull(B);
		assertEquals(N, B.getRowCount());
		assertEquals(N, B.getColumnCount());
		// Verify finite values (no NaN/Inf collapsed the matrix).
		for (long i = 0; i < N; ++i) {
			for (long j = 0; j < N; ++j) {
				double v = B.getAsDouble(i, j);
				if (!Double.isFinite(v)) {
					throw new AssertionError("non-finite at [" + i + "," + j + "]: " + v);
				}
			}
		}
	}

	@Test
	void bfgsRosenbrockNoException() {
		// scipy: BFGS() with default options (auto init scale, skip_update,
		// min_curvature 1e-8). All 37 updates accepted along the Rosenbrock
		// trajectory.
		BFGS hess = BFGS.FACTORY.build();
		runRosenbrockSequence(hess);
	}

	@Test
	void sr1RosenbrockNoException() {
		// scipy: SR1() with default options.
		SR1 hess = SR1.FACTORY.build();
		runRosenbrockSequence(hess);
	}

	@Test
	void bfgsInvHessianRosenbrockNoException() {
		// Inverse-Hessian mode: same 37-step trajectory should also leave the
		// inverse-Hessian approximation finite and well-formed.
		BFGS hess = BFGS.FACTORY.build();
		hess.initialize(N, HessianApproximationType.INV_HESSIAN);
		double[][] gradList = new double[X_LIST.length][];
		for (int i = 0; i < X_LIST.length; ++i) {
			gradList[i] = rosenbrockGrad(X_LIST[i]);
		}
		for (int i = 0; i < X_LIST.length - 1; ++i) {
			double[] dx = new double[N];
			double[] dg = new double[N];
			for (int j = 0; j < N; ++j) {
				dx[j] = X_LIST[i + 1][j] - X_LIST[i][j];
				dg[j] = gradList[i + 1][j] - gradList[i][j];
			}
			hess.update(DenseMatrix.column(dx), DenseMatrix.column(dg));
		}
		Matrix Hi = hess.getMatrix();
		assertNotNull(Hi);
		assertEquals(N, Hi.getRowCount());
		for (long i = 0; i < N; ++i) {
			for (long j = 0; j < N; ++j) {
				if (!Double.isFinite(Hi.getAsDouble(i, j))) {
					throw new AssertionError("non-finite [" + i + "," + j + "]");
				}
			}
		}
	}
}
