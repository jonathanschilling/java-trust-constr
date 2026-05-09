package de.labathome.optimization.integration;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;

/**
 * Distribution of electrons on a unit sphere — COPS problem #2 ("Thomson
 * problem"), via scipy's {@code test_minimize_constrained.py::Elec}.
 *
 * <p>For {@code n_electrons} on a sphere, minimize total electrostatic
 * energy subject to each electron lying on or inside the sphere. Scaled-down
 * here to {@code n_electrons = 2}: minimum places the electrons at
 * antipodes; total energy is 1/2 (one inverse-distance pair, distance 2).
 *
 * <p>This is the most algorithmically demanding integration test in the
 * suite: 6 variables, 2 nonlinear inequality constraints, dense Hessians,
 * cold start far from the optimum.
 */
class TestElec {

	private static final int N = 2;

	private static double[][] coordDeltas(double[] xc, double[] yc, double[] zc, int axis) {
		// dx[i][j] = xc[i] - xc[j]
		double[][] d = new double[N][N];
		double[] arr = (axis == 0) ? xc : (axis == 1) ? yc : zc;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				d[i][j] = arr[i] - arr[j];
			}
		}
		return d;
	}

	private static double elecObjective(Matrix v) {
		double[] xc = new double[N];
		double[] yc = new double[N];
		double[] zc = new double[N];
		for (int i = 0; i < N; ++i) {
			xc[i] = v.getAsDouble(i, 0);
			yc[i] = v.getAsDouble(N + i, 0);
			zc[i] = v.getAsDouble(2 * N + i, 0);
		}
		double sum = 0.0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i == j) continue;
				double dx = xc[i] - xc[j];
				double dy = yc[i] - yc[j];
				double dz = zc[i] - zc[j];
				sum += 1.0 / Math.sqrt(dx * dx + dy * dy + dz * dz);
			}
		}
		return 0.5 * sum;
	}

	private static Matrix elecGrad(Matrix v) {
		double[] xc = new double[N];
		double[] yc = new double[N];
		double[] zc = new double[N];
		for (int i = 0; i < N; ++i) {
			xc[i] = v.getAsDouble(i, 0);
			yc[i] = v.getAsDouble(N + i, 0);
			zc[i] = v.getAsDouble(2 * N + i, 0);
		}
		double[][] dx = coordDeltas(xc, yc, zc, 0);
		double[][] dy = coordDeltas(xc, yc, zc, 1);
		double[][] dz = coordDeltas(xc, yc, zc, 2);
		double[] gx = new double[N];
		double[] gy = new double[N];
		double[] gz = new double[N];
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i == j) continue;
				double r2 = dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j] + dz[i][j] * dz[i][j];
				double dm3 = 1.0 / (r2 * Math.sqrt(r2));
				gx[i] -= dx[i][j] * dm3;
				gy[i] -= dy[i][j] * dm3;
				gz[i] -= dz[i][j] * dm3;
			}
		}
		double[] g = new double[3 * N];
		System.arraycopy(gx, 0, g, 0, N);
		System.arraycopy(gy, 0, g, N, N);
		System.arraycopy(gz, 0, g, 2 * N, N);
		return Matrix.Factory.linkToArray(g);
	}

	private static Matrix elecHess(Matrix v) {
		// Closed-form objective Hessian: 3N x 3N. Implementation mirrors scipy's
		// _Elec.hess: build six N x N blocks Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, then
		// stack as the symmetric block matrix [[Hxx, Hxy, Hxz], [Hxy^T, Hyy, Hyz], …].
		// Diagonals of each block are filled from row-sum negation.
		double[] xc = new double[N];
		double[] yc = new double[N];
		double[] zc = new double[N];
		for (int i = 0; i < N; ++i) {
			xc[i] = v.getAsDouble(i, 0);
			yc[i] = v.getAsDouble(N + i, 0);
			zc[i] = v.getAsDouble(2 * N + i, 0);
		}
		double[][] dx = coordDeltas(xc, yc, zc, 0);
		double[][] dy = coordDeltas(xc, yc, zc, 1);
		double[][] dz = coordDeltas(xc, yc, zc, 2);
		double[][] dm3 = new double[N][N];
		double[][] dm5 = new double[N][N];
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i == j) continue;
				double r2 = dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j] + dz[i][j] * dz[i][j];
				double r = Math.sqrt(r2);
				dm3[i][j] = 1.0 / (r2 * r);
				dm5[i][j] = 1.0 / (r2 * r2 * r);
			}
		}
		double[][] Hxx = blockEntry(dx, dx, dm3, dm5, true);
		double[][] Hyy = blockEntry(dy, dy, dm3, dm5, true);
		double[][] Hzz = blockEntry(dz, dz, dm3, dm5, true);
		double[][] Hxy = blockEntry(dx, dy, dm3, dm5, false);
		double[][] Hxz = blockEntry(dx, dz, dm3, dm5, false);
		double[][] Hyz = blockEntry(dy, dz, dm3, dm5, false);
		double[][] H = new double[3 * N][3 * N];
		copyBlock(H, Hxx, 0, 0);
		copyBlock(H, Hxy, 0, N);
		copyBlock(H, Hxz, 0, 2 * N);
		copyBlock(H, Hxy, N, 0);
		copyBlock(H, Hyy, N, N);
		copyBlock(H, Hyz, N, 2 * N);
		copyBlock(H, Hxz, 2 * N, 0);
		copyBlock(H, Hyz, 2 * N, N);
		copyBlock(H, Hzz, 2 * N, 2 * N);
		return Matrix.Factory.linkToArray(H);
	}

	private static double[][] blockEntry(double[][] dA, double[][] dB,
			double[][] dm3, double[][] dm5, boolean diagonalContribution) {
		double[][] block = new double[N][N];
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				if (i == j) continue;
				double off = -3.0 * dA[i][j] * dB[i][j] * dm5[i][j];
				block[i][j] = diagonalContribution ? (dm3[i][j] + off) : off;
			}
		}
		// Diagonal: -sum of off-diagonal entries in row.
		for (int i = 0; i < N; ++i) {
			double s = 0.0;
			for (int j = 0; j < N; ++j) {
				if (i != j) s += block[i][j];
			}
			block[i][i] = -s;
		}
		return block;
	}

	private static void copyBlock(double[][] dst, double[][] src, int rowOff, int colOff) {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				dst[rowOff + i][colOff + j] = src[i][j];
			}
		}
	}

	@Test
	void twoElectronsOnSphere() {
		// Constraint: each electron lies on or inside the sphere — c[i] = x[i]^2+y[i]^2+z[i]^2-1 <= 0.
		Function<Matrix, Matrix> cf = v -> {
			double[] out = new double[N];
			for (int i = 0; i < N; ++i) {
				double xi = v.getAsDouble(i, 0);
				double yi = v.getAsDouble(N + i, 0);
				double zi = v.getAsDouble(2 * N + i, 0);
				out[i] = xi * xi + yi * yi + zi * zi - 1.0;
			}
			return Matrix.Factory.linkToArray(out);
		};
		Function<Matrix, Matrix> cj = v -> {
			double[][] J = new double[N][3 * N];
			for (int i = 0; i < N; ++i) {
				J[i][i] = 2 * v.getAsDouble(i, 0);
				J[i][N + i] = 2 * v.getAsDouble(N + i, 0);
				J[i][2 * N + i] = 2 * v.getAsDouble(2 * N + i, 0);
			}
			return Matrix.Factory.linkToArray(J);
		};
		// Constraint Hessian-of-Lagrangian: sum_i lambda[i] * H_{c_i}; H_{c_i} = 2 * E_{ii}
		// (the matrix with 1 at positions (i,i), (N+i, N+i), (2N+i, 2N+i)). Sum yields a
		// block-diagonal 2*diag(lambda, lambda, lambda).
		BiFunction<Matrix, Matrix, Matrix> ch = (v, lam) -> {
			double[][] H = new double[3 * N][3 * N];
			for (int i = 0; i < N; ++i) {
				double l2 = 2.0 * lam.getAsDouble(i, 0);
				H[i][i] = l2;
				H[N + i][N + i] = l2;
				H[2 * N + i][2 * N + i] = l2;
			}
			return Matrix.Factory.linkToArray(H);
		};
		NonlinearConstraint c = new NonlinearConstraint(cf, cj, ch,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {0.0, 0.0}, null);

		// Match scipy's seed-0 RandomState initial point (computed offline).
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {
				-0.76141812, -0.20841074, -0.24113917, -0.93761493, 0.60174275, 0.27828619
		});
		OptimizeResult r = MinimizeTrustConstr.minimize(
				TestElec::elecObjective, TestElec::elecGrad, TestElec::elecHess,
				x0, c, 2000, 1.0e-8, 1.0e-8);

		// The optimum places the two electrons at antipodes, energy = 0.5.
		Assertions.assertEquals(0.5, r.fun, 1.0e-3);

		// Check antipodal constraint: x[i] ≈ -x[j] for some matching, or coords sum to ~0.
		double sumX = r.x.getAsDouble(0, 0) + r.x.getAsDouble(1, 0);
		double sumY = r.x.getAsDouble(2, 0) + r.x.getAsDouble(3, 0);
		double sumZ = r.x.getAsDouble(4, 0) + r.x.getAsDouble(5, 0);
		Assertions.assertEquals(0.0, sumX, 1.0e-3);
		Assertions.assertEquals(0.0, sumY, 1.0e-3);
		Assertions.assertEquals(0.0, sumZ, 1.0e-3);

		// Each electron on the unit sphere.
		for (int i = 0; i < N; ++i) {
			double xi = r.x.getAsDouble(i, 0);
			double yi = r.x.getAsDouble(N + i, 0);
			double zi = r.x.getAsDouble(2 * N + i, 0);
			double rNorm = xi * xi + yi * yi + zi * zi;
			Assertions.assertEquals(1.0, rNorm, 1.0e-3,
					"Electron " + i + " not on sphere; r^2=" + rNorm);
		}
	}
}
