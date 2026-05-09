package org.scipy.optimize.minimize;

import dev.ludovic.netlib.blas.BLAS;

import org.ujmp.core.Matrix;

/**
 * Symmetric-rank-1 Hessian update strategy.
 *
 * The update is based on the description in [1], p.144-146.
 *
 * @see [1] Nocedal, Jorge, and Stephen J. Wright
 *          "Numerical optimization"
 *          Second Edition (2006)
 */
public class SR1 extends FullHessianUpdateStrategy {

	public static class SR1Factory extends FullHessianUpdateStrategyFactory {

		private double minDenominator;
		private boolean hasMinDenominator;

		private SR1Factory() {
			super();

			minDenominator = Double.NaN;
			hasMinDenominator = false;
		}

		/**
		 * This number, scaled by a normalization factor,
         * defines the minimum denominator magnitude allowed
         * in the update. When the condition is violated we skip
         * the update. By default uses ``1e-8``.
         *
		 * @param minCurvature
		 * @return
		 */
		public SR1Factory minDenominator(double minDenominator) {
			this.minDenominator = minDenominator;
			this.hasMinDenominator = true;
			return this;
		}

		public SR1 build() {
			if (!hasMinDenominator) {
				minDenominator = 1.0e-8;
				hasMinDenominator = true;
			}

			return new SR1(initialScaleAuto, initialScale, minDenominator);
		}
	};

	public static final SR1Factory FACTORY;
	static {
		FACTORY = new SR1Factory();
	}

	private static final BLAS BLAS_INSTANCE = BLAS.getInstance();

	private final double minDenominator;

	protected SR1(boolean initialScaleAuto, double initialScale, double minDenominator) {
		super(initialScaleAuto, initialScale);

		this.minDenominator = minDenominator;
	}

	@Override
	void updateImplementation(Matrix deltaX, Matrix deltaG) {

		// Auxiliary variables w and z
		Matrix w;
		Matrix z;
		switch (approxType) {
		case HESSIAN:
			w = deltaX;
			z = deltaG;
			break;
		case INV_HESSIAN:
			w = deltaG;
			z = deltaX;
			break;
		default:
			throw new RuntimeException("not implemented");
		}

		// Do some common operations
		Object args = null; // compatibility with extra args for user-defined Hessian
		Matrix Mw = this.apply(w, args);
		Matrix zMinusMw = z.minus(Mw);
		double denominator = w.transpose().mtimes(zMinusMw).doubleValue();

		// If the denominator is too small we just skip the update.
		if (Math.abs(denominator) < minDenominator * w.norm2() * zMinusMw.norm2()) {
			return;
		}

		// Update matrix via BLAS dsyr: M += (1/denominator) * (z - Mw)(z - Mw)^T.
		int N = (int) n;
		double[] zmwArr = colVectorToArray(zMinusMw, N);
		switch (approxType) {
		case HESSIAN: {
			double[] flat = symMatrixToFlat(B, N);
			BLAS_INSTANCE.dsyr("U", N, 1.0 / denominator, zmwArr, 1, flat, N);
			mirrorSymmetric(flat, N);
			B = flatToMatrix(flat, N);
			break;
		}
		case INV_HESSIAN: {
			double[] flat = symMatrixToFlat(H, N);
			BLAS_INSTANCE.dsyr("U", N, 1.0 / denominator, zmwArr, 1, flat, N);
			mirrorSymmetric(flat, N);
			H = flatToMatrix(flat, N);
			break;
		}
		default:
			throw new RuntimeException("not implemented");
		}
	}

	private static double[] colVectorToArray(Matrix v, int n) {
		double[] out = new double[n];
		for (int i = 0; i < n; ++i) {
			out[i] = v.getAsDouble(i, 0);
		}
		return out;
	}

	private static double[] symMatrixToFlat(Matrix m, int n) {
		double[] flat = new double[n * n];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				flat[i * n + j] = m.getAsDouble(i, j);
			}
		}
		return flat;
	}

	private static void mirrorSymmetric(double[] flat, int n) {
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				flat[i * n + j] = flat[j * n + i];
			}
		}
	}

	private static Matrix flatToMatrix(double[] flat, int n) {
		double[][] arr = new double[n][n];
		for (int i = 0; i < n; ++i) {
			System.arraycopy(flat, i * n, arr[i], 0, n);
		}
		return Matrix.Factory.linkToArray(arr);
	}
}
