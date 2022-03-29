package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

/**
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
 *
 * The update is based on the description in [1], p.140.
 *
 * @see [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
 *          Second Edition (2006).
 */
public class BFGS extends FullHessianUpdateStrategy {

	public static class BFGSFactory extends FullHessianUpdateStrategyFactory {

		private ExceptionStrategy exceptionStrategy;

		private double minCurvature;
		private boolean hasMinCurvature;

		private BFGSFactory() {
			super();

			exceptionStrategy = ExceptionStrategy.SKIP_UPDATE;

			minCurvature = Double.NaN;
			hasMinCurvature = false;
		}

		/**
		 * Define how to proceed when the curvature condition is violated.
		 * Set it to 'skip_update' to just skip the update. Or,
		 *                          alternatively, set it to 'damp_update' to
		 *                          interpolate between the actual BFGS result and the
		 *                          unmodified matrix. Both exceptions strategies are
		 *                          explained in [1], p.536-537.
		 *
		 * @param exceptionStrategy
		 * @return
		 */
		public BFGSFactory exceptionStrategy(ExceptionStrategy exceptionStrategy) {
			this.exceptionStrategy = exceptionStrategy;
			return this;
		}

		/**
		 * This number, scaled by a normalization factor, defines the
		 * minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
		 * unaffected by the exception strategy. By default is equal to
		 * 1e-8 when ``exception_strategy = 'skip_update'`` and equal
		 * to 0.2 when ``exception_strategy = 'damp_update'``.
		 *
		 * @param minCurvature
		 * @return
		 */
		public BFGSFactory minCurvature(double minCurvature) {
			this.minCurvature = minCurvature;
			this.hasMinCurvature = true;
			return this;
		}

		public BFGS build() {
			switch(exceptionStrategy) {
			case SKIP_UPDATE:
				if (!hasMinCurvature) {
					minCurvature = 1.0e-8;
					hasMinCurvature = true;
				}
				break;
			case DAMP_UPDATE:
				if (!hasMinCurvature) {
					minCurvature = 0.2;
					hasMinCurvature = true;
				}
				break;
			default:
				throw new RuntimeException("not implemented");
			}

			return new BFGS(initialScaleAuto, initialScale, exceptionStrategy, minCurvature);
		}
	};

	public static final BFGSFactory FACTORY;
	static {
		FACTORY = new BFGSFactory();
	}

	private ExceptionStrategy exceptionStrategy;
	private double minCurvature;

	private BFGS(boolean initialScaleAuto, double initialScale, ExceptionStrategy exceptionStrategy, double minCurvature) {
		super(initialScaleAuto, initialScale);

		this.exceptionStrategy = exceptionStrategy;
		this.minCurvature = minCurvature;
	}

	/**
	 * Update the inverse Hessian matrix.
     *
     * BFGS update using the formula:
     *
     *     ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
     *              - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``
     *
     * where ``s = delta_x`` and ``y = delta_grad``. This formula is
     * equivalent to (6.17) in [1] written in a more efficient way
     * for implementation.
     *
     * References
     * ----------
     * .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
     *        Second Edition (2006).
	 * @param ys
	 * @param Hy
	 * @param yHy
	 * @param s
	 */
	private void updateInverseHessian(double ys, Matrix Hy, double yHy, Matrix s) {

		// TODO: use _syr2 from BLAS
		// This is the second row in above equation.
		double alpha = -1.0/ys;
		H = H.plus( ( Hy.mtimes(s.transpose()).plus( s.mtimes(Hy.transpose()) ) ).times(alpha) );

		// TODO: use _syr from BLAS
		// This is the first row in above equation.
		double alpha2 = (ys + yHy)/(ys*ys);
		H = H.plus( s.mtimes(s.transpose()).times(alpha2) );
	}

	/**
	 * Update the Hessian matrix.
     *
     * BFGS update using the formula:
     *
     *     ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``
     *
     * where ``s`` is short for ``delta_x`` and ``y`` is short
     * for ``delta_grad``. Formula (6.19) in [1].
     *
     * References
     * ----------
     * .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
     *        Second Edition (2006).
	 * @param ys
	 * @param Bs
	 * @param sBs
	 * @param y
	 */
	private void updateHessian(double ys, Matrix Bs, double sBs, Matrix y) {

		// TODO: use _syr from BLAS
		// This is the second term in above equation.
		B = B.plus( y.mtimes(y.transpose()).times(1.0/ys) );

		// TODO: use _syr from BLAS
		// This is the first term in above equation.
		B = B.plus( Bs.mtimes(Bs.transpose()).times(-1.0/sBs) );
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
		double wz = w.transpose().mtimes(z).doubleValue();
		Matrix Mw = this.dot(w);
		double wMw = Mw.mtimes(w).doubleValue();

		// Guarantee that wMw > 0 by reinitializing matrix.
        // While this is always true in exact arithmetics,
        // indefinite matrix may appear due to roundoff errors.
		if (wMw <= 0.0) {
			double scale = autoScale(deltaX, deltaG);

			// Reinitialize matrix
			switch (approxType) {
			case HESSIAN:
				B = Matrix.Factory.eye(n, n).times(scale);
				break;
			case INV_HESSIAN:
				H = Matrix.Factory.eye(n, n).times(scale);
				break;
			default:
				throw new RuntimeException("not implemented");
			}

			// Do common operations for new matrix
			Mw = this.dot(w);
			wMw = Mw.mtimes(w).doubleValue();
		}

		// Check if curvature condition is violated
		if (wz <= minCurvature * wMw) {

			switch (exceptionStrategy) {
			case SKIP_UPDATE:
				// If the option 'skip_update' is set
	            // we just skip the update when the condion
	            // is violated.
				return;
			case DAMP_UPDATE:
				// If the option 'damp_update' is set we
	            // interpolate between the actual BFGS
	            // result and the unmodified matrix.
				double updateFactor = (1.0 - minCurvature) / (1.0 - wz/wMw);
				z = z.times(updateFactor).plus( Mw.times(1.0-updateFactor) );
				wz = w.transpose().mtimes(z).doubleValue();
				break;
			default:
				throw new RuntimeException("not implemented");
			}
		}

		// Update matrix
		switch (approxType) {
		case HESSIAN:
			updateHessian(wz, Mw, wMw, z);
			break;
		case INV_HESSIAN:
			updateInverseHessian(wz, Mw, wMw, z);
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}
}
