package org.scipy.optimize.minimize;

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

	private double minDenominator;

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
		Matrix Mw = this.dot(w);
		Matrix zMinusMw = z.minus(Mw);
		double denominator = w.transpose().mtimes(zMinusMw).doubleValue();

		// If the denominator is too small we just skip the update.
		if (Math.abs(denominator) < minDenominator * w.norm2() * zMinusMw.norm2()) {
			return;
		}

		// Update matrix
		switch (approxType) {
		case HESSIAN:
			// TODO: use _syr from BLAS
			B = B.plus( zMinusMw.mtimes(zMinusMw.transpose()).times(1/denominator) );
			break;
		case INV_HESSIAN:
			// TODO: use _syr from BLAS
			H = H.plus( zMinusMw.mtimes(zMinusMw.transpose()).times(1/denominator) );
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}
}
