package de.labathome.trustconstr;

import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.LinAlg;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Symmetric-rank-1 (SR1) Hessian update strategy.
 *
 * <p>The update follows Nocedal &amp; Wright, <i>Numerical Optimization</i>,
 * 2nd ed. (2006), pp.144-146.
 */
public class SR1 extends FullHessianUpdateStrategy {

	/** Builder for {@link SR1}; configure via {@link SR1#FACTORY}. */
	public static class SR1Factory extends FullHessianUpdateStrategyFactory {

		private double minDenominator;
		private boolean hasMinDenominator;

		private SR1Factory() {
			super();

			minDenominator = Double.NaN;
			hasMinDenominator = false;
		}

		/**
		 * Set the minimum denominator magnitude allowed in the update,
		 * scaled by a normalization factor. When the condition is violated
		 * the update is skipped. Default {@code 1e-8}.
		 *
		 * @param minDenominator denominator-magnitude threshold
		 * @return this factory, for chaining
		 */
		public SR1Factory minDenominator(double minDenominator) {
			this.minDenominator = minDenominator;
			this.hasMinDenominator = true;
			return this;
		}

		/**
		 * Build a configured {@link SR1} instance.
		 *
		 * @return a fresh {@link SR1} with the configured options
		 */
		public SR1 build() {
			if (!hasMinDenominator) {
				minDenominator = 1.0e-8;
				hasMinDenominator = true;
			}

			return new SR1(initialScaleAuto, initialScale, minDenominator);
		}
	};

	/** Default SR1 factory; configure with chained setters then call {@link SR1Factory#build()}. */
	public static final SR1Factory FACTORY;
	static {
		FACTORY = new SR1Factory();
	}

	private final double minDenominator;

	/**
	 * @param initialScaleAuto whether the initial scale is auto-derived
	 * @param initialScale     user-specified initial scale (ignored when auto)
	 * @param minDenominator   denominator-magnitude threshold below which updates are skipped
	 */
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
		double[] zmwArr = colVectorToArray(zMinusMw);
		switch (approxType) {
		case HESSIAN:
			LinAlg.syr(B, 1.0 / denominator, zmwArr);
			break;
		case INV_HESSIAN:
			LinAlg.syr(H, 1.0 / denominator, zmwArr);
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	private static double[] colVectorToArray(Matrix v) {
		if (v instanceof DenseMatrix d) {
			return d.toColumnArray();
		}
		int n = (int) v.getRowCount();
		double[] out = new double[n];
		for (int i = 0; i < n; ++i) {
			out[i] = v.getAsDouble(i, 0);
		}
		return out;
	}
}
