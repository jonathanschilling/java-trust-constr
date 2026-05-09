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
package de.labathome.trustconstr.demo;

import java.util.function.Function;

import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.SR1;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.OptimizeResult;

/**
 * Demo: spectral condensation of a 2D closed curve.
 *
 * <p>The shape is a stellarator-style 2D boundary parameterised as
 * <pre>
 *     x(theta) = sum_{m=1..M} X_m cos(m theta)
 *     y(theta) = sum_{m=1..M} Y_m sin(m theta)
 * </pre>
 * sampled on the half-period {@code [0, pi]} at {@code theta_i = i * 2*pi/N}
 * for {@code i = 0..N/2}. We then look for Fourier coefficients
 * {@code (X_m, Y_m)} (with {@code M > M_target = 3}) and a reparametrisation
 * <pre>
 *     alpha_i = theta_i + lambda(theta_i),
 *     lambda(theta) = sum_{l=1..L} L_l sin(l theta)
 * </pre>
 * such that the new curve passes exactly through the original sample points
 * <pre>
 *     x(alpha_i) = x_i,  y(alpha_i) = y_i   for every sample i,
 * </pre>
 * while minimising the spectral width
 * <pre>
 *     M(p,q) = (sum_m m^{p+q} S_m) / (sum_m m^p S_m),  S_m = X_m^2 + Y_m^2,
 * </pre>
 * with {@code p = 4, q = 1}.
 *
 * <p>The trivial guess (X_m = X_m^target, Y_m = Y_m^target, lambda = 0)
 * trivially satisfies {@code W_c = 0} and provides a baseline spectral
 * width. The optimiser then redistributes coefficient energy across the
 * higher modes ({@code m > M_target}) by reparametrising via {@code lambda},
 * driving {@code M(p,q)} below the baseline while keeping {@code W_c} at
 * machine zero.
 *
 * <p>This is the Hirshman-Meier spectral condensation problem that arises
 * in stellarator boundary descriptions.
 *
 * <p>Constraint dimensioning (with {@code N = 16}):
 * <ul>
 *   <li>{@code N_samples = N/2 + 1 = 9} sample points on {@code [0, pi]}.</li>
 *   <li>{@code L = N/2 - 1 = 7} lambda modes -- exactly one mode per
 *       interior sample, so lambda can match any prescribed values at the
 *       interior samples (the endpoints are pinned by the sin-only ansatz
 *       since {@code lambda(0) = lambda(pi) = 0} automatically).</li>
 *   <li>{@code M = 6 > M_target = 3} curve modes for both x and y.</li>
 *   <li>Free parameters: {@code 2M + L = 19}.</li>
 *   <li>Effective equality constraints: {@code 2*N_samples - 2 = 16}.
 *       (The y-constraint at {@code theta = 0} and {@code theta = pi} is
 *       trivially {@code 0 = 0} for a sin-only series and is dropped to
 *       keep the constraint Jacobian full row rank.)</li>
 *   <li>Excess DOF for spectral minimisation: {@code 19 - 16 = 3}.</li>
 * </ul>
 */
public final class SpectralCondensation {

	/** Full-period sample count (must be even). */
	private static final int N = 30;
	/** Sample count on {@code [0, pi]}. */
	private static final int N_SAMPLES = N / 2 + 1;
	/** Lambda mode count. */
	private static final int L = N / 2 - 1;
	/**
	 * Curve mode count. Must satisfy {@code M > M_TARGET} for the problem to
	 * be more than a trivial round-trip, and {@code 2*M + L > 2*N_SAMPLES - 2}
	 * for the equality system to be under-determined enough to give the
	 * optimiser any freedom to redistribute spectral content. With
	 * {@code N = 30, L = 14}, the {@code M = 10} below leaves 4 excess DOF.
	 */
	private static final int M = 10;
	/** Target mode count (length of the supplied target Fourier vectors). */
	private static final int M_TARGET = 3;

	/** Spectral-width exponents. */
	private static final double P_EXP = 4.0;
	private static final double Q_EXP = 1.0;

	/** Target curve coefficients (m = 1, 2, 3). */
	private static final double[] X_COS_TARGET = { 1.042, 0.502, -0.0389 };
	private static final double[] Y_SIN_TARGET = { 1.339, 0.296, -0.0176 };

	/**
	 * Parameter packing in the optimiser variable {@code p}:
	 * <ul>
	 *   <li>{@code p[0..M-1]}        : {@code X_m} for {@code m = 1..M}</li>
	 *   <li>{@code p[M..2M-1]}       : {@code Y_m} for {@code m = 1..M}</li>
	 *   <li>{@code p[2M..2M+L-1]}    : {@code L_l} for {@code l = 1..L}</li>
	 * </ul>
	 */
	private static final int N_PARAMS = 2 * M + L;

	/** Sample positions in {@code [0, pi]}. */
	private static final double[] THETA = new double[N_SAMPLES];
	static {
		for (int i = 0; i < N_SAMPLES; ++i) {
			THETA[i] = i * 2.0 * Math.PI / N;
		}
	}

	/** Target sample points {@code (x_i, y_i)} computed from the target coefficients. */
	private static final double[] X_TARGET = new double[N_SAMPLES];
	private static final double[] Y_TARGET = new double[N_SAMPLES];
	static {
		for (int i = 0; i < N_SAMPLES; ++i) {
			X_TARGET[i] = curveX(X_COS_TARGET, THETA[i]);
			Y_TARGET[i] = curveY(Y_SIN_TARGET, THETA[i]);
		}
	}

	/**
	 * Active constraint rows. Encoded as integers in {@code [0, 2*N_SAMPLES)}:
	 * {@code [0, N_SAMPLES)} = x-constraint at sample i;
	 * {@code [N_SAMPLES, 2*N_SAMPLES)} = y-constraint at sample {@code i - N_SAMPLES}.
	 *
	 * <p>The y-constraint at {@code i = 0} and {@code i = N_SAMPLES - 1} is
	 * always zero (the y series is sin-only; alpha is forced to 0 or pi
	 * there because lambda vanishes), so those rows are dropped.
	 */
	private static final int[] ACTIVE_ROWS = activeRows();

	private static int[] activeRows() {
		int[] rows = new int[2 * N_SAMPLES - 2];
		int k = 0;
		for (int i = 0; i < N_SAMPLES; ++i) {
			rows[k++] = i;
		}
		for (int i = 1; i < N_SAMPLES - 1; ++i) {
			rows[k++] = N_SAMPLES + i;
		}
		return rows;
	}

	private static final int N_ROWS = ACTIVE_ROWS.length;

	private SpectralCondensation() {}

	// -- curve / lambda evaluators ---------------------------------------

	private static double curveX(double[] xcos, double alpha) {
		double s = 0.0;
		for (int m = 0; m < xcos.length; ++m) {
			s += xcos[m] * Math.cos((m + 1) * alpha);
		}
		return s;
	}

	private static double curveY(double[] ysin, double alpha) {
		double s = 0.0;
		for (int m = 0; m < ysin.length; ++m) {
			s += ysin[m] * Math.sin((m + 1) * alpha);
		}
		return s;
	}

	private static double lambda(double[] lsin, double theta) {
		double s = 0.0;
		for (int l = 0; l < lsin.length; ++l) {
			s += lsin[l] * Math.sin((l + 1) * theta);
		}
		return s;
	}

	// -- packing / unpacking ---------------------------------------------

	private static final class Params {
		final double[] X = new double[M];
		final double[] Y = new double[M];
		final double[] Lc = new double[L];
	}

	private static Params unpack(Matrix p) {
		Params out = new Params();
		for (int m = 0; m < M; ++m) out.X[m] = p.getAsDouble(m, 0);
		for (int m = 0; m < M; ++m) out.Y[m] = p.getAsDouble(M + m, 0);
		for (int l = 0; l < L; ++l) out.Lc[l] = p.getAsDouble(2 * M + l, 0);
		return out;
	}

	/**
	 * Pack {@code (X, Y, lambda)} coefficients into the optimiser parameter
	 * vector. Inputs shorter than the configured length are zero-padded.
	 */
	private static Matrix pack(double[] X, double[] Y, double[] Lc) {
		double[] data = new double[N_PARAMS];
		for (int m = 0; m < M && m < X.length; ++m)  data[m] = X[m];
		for (int m = 0; m < M && m < Y.length; ++m)  data[M + m] = Y[m];
		for (int l = 0; l < L && l < Lc.length; ++l) data[2 * M + l] = Lc[l];
		return DenseMatrix.column(data);
	}

	// -- spectral width and gradient -------------------------------------

	private static double spectralWidth(Params p) {
		double num = 0.0;
		double den = 0.0;
		for (int m = 0; m < M; ++m) {
			double Sm = p.X[m] * p.X[m] + p.Y[m] * p.Y[m];
			double mp1 = m + 1;
			double mPow = Math.pow(mp1, P_EXP);
			num += mPow * Math.pow(mp1, Q_EXP) * Sm;
			den += mPow * Sm;
		}
		return num / den;
	}

	/**
	 * Gradient of the spectral width w.r.t. the full parameter vector.
	 * <pre>
	 *     dM/dX_m = (2 X_m m^p / D) (m^q - M),
	 *     dM/dY_m = (2 Y_m m^p / D) (m^q - M),
	 *     dM/dL_l = 0.
	 * </pre>
	 */
	private static Matrix spectralWidthGrad(Matrix q) {
		Params p = unpack(q);
		double num = 0.0;
		double den = 0.0;
		for (int m = 0; m < M; ++m) {
			double Sm = p.X[m] * p.X[m] + p.Y[m] * p.Y[m];
			double mp1 = m + 1;
			double mPow = Math.pow(mp1, P_EXP);
			num += mPow * Math.pow(mp1, Q_EXP) * Sm;
			den += mPow * Sm;
		}
		double Mpq = num / den;

		double[] g = new double[N_PARAMS];
		for (int m = 0; m < M; ++m) {
			double mp1 = m + 1;
			double factor = 2.0 * Math.pow(mp1, P_EXP)
					* (Math.pow(mp1, Q_EXP) - Mpq) / den;
			g[m]     = factor * p.X[m];
			g[M + m] = factor * p.Y[m];
		}
		return DenseMatrix.column(g);
	}

	// -- W_c equality constraint and Jacobian ----------------------------

	/**
	 * Per-sample residual vector for the W_c equality constraint, with
	 * trivially-zero y-rows at theta=0 and theta=pi dropped (16 rows for
	 * {@code N = 16}). Driving every entry to zero is equivalent to
	 * {@code W_c = sum (residual^2) = 0}.
	 */
	private static Matrix constraintFun(Matrix p) {
		Params par = unpack(p);
		double[] vals = new double[N_ROWS];
		for (int k = 0; k < N_ROWS; ++k) {
			int row = ACTIVE_ROWS[k];
			boolean isX = row < N_SAMPLES;
			int i = isX ? row : row - N_SAMPLES;
			double th = THETA[i];
			double alpha_i = th + lambda(par.Lc, th);
			vals[k] = isX
					? curveX(par.X, alpha_i) - X_TARGET[i]
					: curveY(par.Y, alpha_i) - Y_TARGET[i];
		}
		return DenseMatrix.column(vals);
	}

	/**
	 * Analytic Jacobian of {@link #constraintFun(Matrix)} (shape {@code N_ROWS x N_PARAMS}).
	 *
	 * <p>For an x-constraint row at sample i:
	 * <pre>
	 *     d c_xi / d X_m = cos(m alpha_i),
	 *     d c_xi / d Y_m = 0,
	 *     d c_xi / d L_l = (-sum_m m X_m sin(m alpha_i)) * sin(l theta_i).
	 * </pre>
	 * For a y-constraint row at sample i:
	 * <pre>
	 *     d c_yi / d X_m = 0,
	 *     d c_yi / d Y_m = sin(m alpha_i),
	 *     d c_yi / d L_l = ( sum_m m Y_m cos(m alpha_i)) * sin(l theta_i).
	 * </pre>
	 */
	private static Matrix constraintJac(Matrix p) {
		Params par = unpack(p);
		double[][] J = new double[N_ROWS][N_PARAMS];
		for (int k = 0; k < N_ROWS; ++k) {
			int row = ACTIVE_ROWS[k];
			boolean isX = row < N_SAMPLES;
			int i = isX ? row : row - N_SAMPLES;
			double th = THETA[i];
			double alpha_i = th + lambda(par.Lc, th);

			if (isX) {
				for (int m = 0; m < M; ++m) {
					J[k][m] = Math.cos((m + 1) * alpha_i);
				}
				double dxda = 0.0;
				for (int m = 0; m < M; ++m) {
					dxda -= (m + 1) * par.X[m] * Math.sin((m + 1) * alpha_i);
				}
				for (int l = 0; l < L; ++l) {
					J[k][2 * M + l] = dxda * Math.sin((l + 1) * th);
				}
			} else {
				for (int m = 0; m < M; ++m) {
					J[k][M + m] = Math.sin((m + 1) * alpha_i);
				}
				double dyda = 0.0;
				for (int m = 0; m < M; ++m) {
					dyda += (m + 1) * par.Y[m] * Math.cos((m + 1) * alpha_i);
				}
				for (int l = 0; l < L; ++l) {
					J[k][2 * M + l] = dyda * Math.sin((l + 1) * th);
				}
			}
		}
		return DenseMatrix.fromRows(J);
	}

	/**
	 * Add a small symmetric kick to the lambda coefficients to step off the
	 * degenerate zero-lambda + zero-high-mode start. The first entry is set
	 * to {@code +eps}, the second to {@code -eps}, and the rest stay zero --
	 * just enough perturbation to make the QP subproblem well-conditioned at
	 * the first SQP iteration.
	 */
	private static Matrix nudgeLambda(Matrix x0, double eps) {
		double[] data = new double[N_PARAMS];
		for (int i = 0; i < N_PARAMS; ++i) data[i] = x0.getAsDouble(i, 0);
		if (L >= 1) data[2 * M + 0] += eps;
		if (L >= 2) data[2 * M + 1] -= eps;
		return DenseMatrix.column(data);
	}

	/** Sum of squared residuals: this is the W_c value in the problem statement. */
	private static double curveEnergy(Matrix p) {
		Matrix r = constraintFun(p);
		double s = 0.0;
		for (long i = 0; i < r.getRowCount(); ++i) {
			double v = r.getAsDouble(i, 0);
			s += v * v;
		}
		return s;
	}

	// -- main ------------------------------------------------------------

	public static void main(String[] args) {
		Matrix x0 = pack(X_COS_TARGET, Y_SIN_TARGET, new double[L]);
		Params par0 = unpack(x0);

		double m0 = spectralWidth(par0);
		double wc0 = curveEnergy(x0);

		System.out.printf("Problem dimensions:%n");
		System.out.printf("  N=%d, N_samples=%d, M=%d (target %d), L=%d%n",
				N, N_SAMPLES, M, M_TARGET, L);
		System.out.printf("  free parameters: %d, equality constraints: %d, excess DOF: %d%n",
				N_PARAMS, N_ROWS, N_PARAMS - N_ROWS);
		System.out.println();

		System.out.println("--- trivial initial guess ---");
		System.out.printf("  spectral width M(p=%.0f, q=%.0f) = %.10e%n", P_EXP, Q_EXP, m0);
		System.out.printf("  curve energy W_c              = %.3e%n", wc0);
		System.out.println();

		Function<Matrix, Double> fun = q -> spectralWidth(unpack(q));
		Function<Matrix, Matrix> grad = SpectralCondensation::spectralWidthGrad;

		double[] eqBound = new double[N_ROWS];
		NonlinearConstraint c = new NonlinearConstraint(
				SpectralCondensation::constraintFun,
				SpectralCondensation::constraintJac,
				null,
				eqBound, eqBound, null);

		// The exact zero-lambda + zero-high-mode start is a degenerate point
		// of the spectral-width landscape: at X_m = 0 for m > M_TARGET we have
		// dM/dX_m = 2*X_m*(...) = 0 trivially, and the constraint Jacobian's
		// lambda-columns also vanish at theta=0 and theta=pi. The QP subproblem
		// at that point produces a near-zero step, the trust radius collapses,
		// and the optimiser xtol-terminates without ever leaving x0 (regardless
		// of BFGS vs SR1 -- the issue is the geometry, not the Hessian model).
		// A small deterministic kick on lambda breaks the symmetry; the rest
		// of the optimisation then proceeds normally.
		Matrix xStart = nudgeLambda(x0, 1.0e-3);

		OptimizeResult res = MinimizeTrustConstr.minimize(
				fun, grad, new SR1.SR1Factory().build(), xStart, c,
				/*maxIter=*/ 1000, /*xtol=*/ 1e-14, /*gtol=*/ 1e-12);

		Params parf = unpack(res.x);
		double mf = spectralWidth(parf);
		double wcf = curveEnergy(res.x);

		System.out.println("--- after optimisation ---");
		System.out.printf("  status            = %d (%s)%n", res.status, res.message);
		System.out.printf("  iterations        = %d (CG iters: %d)%n", res.nIter, res.cgIter);
		System.out.printf("  fun evals=%d, jac evals=%d, hess evals=%d%n",
				res.numFunctionEval, res.numJacobianEval, res.numHessianEval);
		System.out.printf("  spectral width    = %.10e   (reduction: %.3fx)%n", mf, m0 / mf);
		System.out.printf("  curve energy W_c  = %.3e%n", wcf);
		System.out.printf("  optimality (||grad L||_inf)  = %.3e%n", res.optimality);
		System.out.printf("  constraint violation         = %.3e%n", res.constraintViolation);
		System.out.println();

		System.out.println("  curve coefficients   m: X_m              Y_m");
		for (int m = 0; m < M; ++m) {
			String tag = (m < M_TARGET) ? "" : "  (added beyond target)";
			System.out.printf("    %2d: %+.10e   %+.10e%s%n", m + 1, parf.X[m], parf.Y[m], tag);
		}
		System.out.println("  lambda coefficients  l: L_l");
		for (int l = 0; l < L; ++l) {
			System.out.printf("    %2d: %+.10e%n", l + 1, parf.Lc[l]);
		}

		// Perturbed-restart probe: rerun from random lambda seeds with a
		// different Hessian update strategy (SR1) to test whether BFGS had
		// merely settled in a local minimum. If every restart lands on the
		// same value (as it does on this problem), that's strong evidence
		// for a global minimum given the geometry of the target curve.
		System.out.println();
		System.out.println("--- perturbed-restart probe (random lambda seed, SR1 strategy) ---");
		java.util.Random rng = new java.util.Random(42L);
		int better = 0;
		double bestM = mf;
		for (int trial = 0; trial < 5; ++trial) {
			double[] Lc = new double[L];
			for (int l = 0; l < L; ++l) Lc[l] = 0.05 * (rng.nextDouble() - 0.5);
			Matrix xs = pack(X_COS_TARGET, Y_SIN_TARGET, Lc);
			OptimizeResult r = MinimizeTrustConstr.minimize(
					fun, grad, new SR1.SR1Factory().build(), xs, c,
					/*maxIter=*/ 1000, /*xtol=*/ 1e-14, /*gtol=*/ 1e-12);
			double sw = spectralWidth(unpack(r.x));
			double wc = curveEnergy(r.x);
			System.out.printf("  trial %d: M=%.10e  W_c=%.2e  status=%d  iters=%d%n",
					trial, sw, wc, r.status, r.nIter);
			if (sw < bestM - 1e-12) { ++better; bestM = sw; }
		}
		System.out.printf("  perturbed runs that beat the deterministic optimum: %d/5  (best M = %.10e)%n",
				better, bestM);
	}
}
