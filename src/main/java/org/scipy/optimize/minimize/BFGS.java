package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.ExceptionStrategy;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.LinAlg;
import org.scipy.optimize.minimize.matrix.Matrix;

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

	private final ExceptionStrategy exceptionStrategy;
	private final double minCurvature;

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
		double[] sArr = colVectorToArray(s);
		double[] HyArr = colVectorToArray(Hy);

		// H += -1/ys * (Hy s^T + s Hy^T) — second row of the BFGS formula.
		LinAlg.syr2(H, -1.0 / ys, HyArr, sArr);
		// H += (ys + yHy)/(ys^2) * s s^T — first row of the BFGS formula.
		LinAlg.syr(H, (ys + yHy) / (ys * ys), sArr);
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
		double[] yArr = colVectorToArray(y);
		double[] BsArr = colVectorToArray(Bs);

		// B += (1/ys) * y y^T — second term.
		LinAlg.syr(B, 1.0 / ys, yArr);
		// B += -(1/sBs) * Bs Bs^T — first term.
		LinAlg.syr(B, -1.0 / sBs, BsArr);
	}

	/**
	 * Copy a column vector ({@code n × 1}) {@code v} into a length-{@code n}
	 * {@code double[]}. When {@code v} is a {@link DenseMatrix} this is a
	 * single {@code data().clone()}; otherwise it falls back to element-wise
	 * extraction via {@link Matrix#getAsDouble(long, long)}.
	 */
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
		double wz = w.transpose().mtimes(z).doubleValue();
		Matrix Mw = this.apply(w, args);
		double wMw = w.transpose().mtimes(Mw).doubleValue();

		// Guarantee that wMw > 0 by reinitializing matrix.
        // While this is always true in exact arithmetics,
        // indefinite matrix may appear due to roundoff errors.
		if (wMw <= 0.0) {
			double scale = autoScale(deltaX, deltaG);

			// Reinitialize matrix
			switch (approxType) {
			case HESSIAN:
				B = DenseMatrix.eye((int) n).times(scale);
				break;
			case INV_HESSIAN:
				H = DenseMatrix.eye((int) n).times(scale);
				break;
			default:
				throw new RuntimeException("not implemented");
			}

			// Do common operations for new matrix
			Mw = this.apply(w, args);
			wMw = w.transpose().mtimes(Mw).doubleValue();
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
