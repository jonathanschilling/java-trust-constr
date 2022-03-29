package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

/**
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
 *
 * min_curvature : float
 *     This number, scaled by a normalization factor, defines the
 *     minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
 *     unaffected by the exception strategy. By default is equal to
 *     1e-8 when ``exception_strategy = 'skip_update'`` and equal
 *     to 0.2 when ``exception_strategy = 'damp_update'``.
 * init_scale : {float, 'auto'}
 *     Matrix scale at first iteration. At the first
 *     iteration the Hessian matrix or its inverse will be initialized
 *     with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
 *     Set it to 'auto' in order to use an automatic heuristic for choosing
 *     the initial scale. The heuristic is described in [1]_, p.143.
 *     By default uses 'auto'.
 *
 * The update is based on the description in [1], p.140.
 *
 * @see [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
 *          Second Edition (2006).
 */
public class BFGS extends FullHessianUpdateStrategy {

	public static class Factory {
		ExceptionStrategy exceptionStrategy;

		double minCurvature;
		boolean hasMinCurvature;

		public Factory() {
			hasMinCurvature = false;
		}

		/**
		 * Define how to proceed when the curvature condition is violated.
		 *
		 * @param exceptionStrategy Set it to 'skip_update' to just skip the update. Or,
		 *                          alternatively, set it to 'damp_update' to
		 *                          interpolate between the actual BFGS result and the
		 *                          unmodified matrix. Both exceptions strategies are
		 *                          explained in [1], p.536-537.
		 * @return
		 */
		public Factory exceptionStrategy(ExceptionStrategy exceptionStrategy) {
			this.exceptionStrategy = exceptionStrategy;
			return this;
		}

		/**
		 * This number, scaled by a normalization factor, defines the
		 * minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
		 * unaffected by the exception strategy. By default is equal to
		 * 1e-8 when ``exception_strategy = 'skip_update'`` and equal
		 * to 0.2 when ``exception_strategy = 'damp_update'``.
		 * @param minCurvature
		 * @return
		 */
		public Factory minCurvature(double minCurvature) {
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

			return new BFGS(exceptionStrategy, minCurvature);
		}
	};

	ExceptionStrategy exceptionStrategy;
	double minCurvature;

	private BFGS(ExceptionStrategy exceptionStrategy, double minCurvature) {
		this.exceptionStrategy = exceptionStrategy;
		this.minCurvature = minCurvature;
	}

	@Override
	void updateImplementation(Matrix deltaX, Matrix deltaG) {


		Matrix wz = null;
		Matrix Mw = null;
		Matrix wMw = null;
		Matrix z = null;






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

	/**
	 * Update the inverse Hessian matrix.
     *
     * BFGS update using the formula:
     *
     *     ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
     *              - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``
     *
     * where ``s = delta_x`` and ``y = delta_grad``. This formula is
     * equivalent to (6.17) in [1]_ written in a more efficient way
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
	private void updateInverseHessian(Matrix ys, Matrix Hy, Matrix yHy, Matrix s) {

	}

	/**
	 * Update the Hessian matrix.
     *
     * BFGS update using the formula:
     *
     *     ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``
     *
     * where ``s`` is short for ``delta_x`` and ``y`` is short
     * for ``delta_grad``. Formula (6.19) in [1]_.
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
	private void updateHessian(Matrix ys, Matrix Bs, Matrix sBs, Matrix y) {

	}


}
