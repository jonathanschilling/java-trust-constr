package org.scipy.optimize.minimize;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

/**
 * Hessian update strategy with full dimensional internal representation.
 */
public abstract class FullHessianUpdateStrategy implements HessianUpdateStrategy {

	private double initialScale;
	private boolean initScaleAuto;

	double scale;

	boolean firstIteration;
	HessianApproximationType approxType;

	/** problem dimension */
	long n;

	/** Hessian */
	Matrix B;

	/** inverse Hessian */
	Matrix H;

	public FullHessianUpdateStrategy() {

		// default: init_scale = 'auto'
		initScaleAuto();

		firstIteration = false;
		approxType = null;
	}

	public FullHessianUpdateStrategy initScaleAuto() {
		this.initialScale = Double.NaN;
		this.initScaleAuto = true;
		return this;
	}

	public FullHessianUpdateStrategy initScale(double initScale) {
		this.initialScale = initScale;
		this.initScaleAuto = false;
		return this;
	}

	@Override
	public void initialize(long n, HessianApproximationType approxType) {
		this.n = n;
		this.approxType = approxType;

		firstIteration = true;

		// Create matrix
		switch (this.approxType) {
		case HESSIAN:
			this.B = Matrix.Factory.eye(n, n);
			break;
		case INV_HESSIAN:
			this.H = Matrix.Factory.eye(n, n);
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	/**
	 * Heuristic to scale matrix at first iteration.
	 * Described in Nocedal and Wright "Numerical Optimization"
	 * p.143 formula (6.20).
	 *
	 * @param deltaX
	 * @param deltaG
	 */
	private double autoScale(Matrix deltaX, Matrix deltaG) {
		double sNorm2 = deltaX.mtimes(deltaX).doubleValue();
		double yNorm2 = deltaG.mtimes(deltaG).doubleValue();

		double ys = Math.abs(deltaG.mtimes(deltaX).doubleValue());

		if (ys == 0.0 || yNorm2 == 0.0 || sNorm2 == 0.0) {
			// fallback to no scaling
			return 1.0;
		}

		switch (approxType) {
		case HESSIAN:
			return yNorm2 / ys;
		case INV_HESSIAN:
			return ys / yNorm2;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	abstract void updateImplementation(Matrix deltaX, Matrix deltaG);

	@Override
	public void update(Matrix deltaX, Matrix deltaG) {

		if (deltaX.normInf() == 0.0) {
			return;
		}

		if (deltaG.normInf() == 0.0) {
			System.out.println("delta_grad == 0.0. Check if the approximated\n" +
					"function is linear. If the function is linear\n" +
					"better results can be obtained by defining the\n" +
					"Hessian as zero instead of using quasi-Newton\n" +
					"approximations.");
		}

		if (firstIteration) {
			// Get user specific scale
			if (initScaleAuto) {
				scale = autoScale(deltaX, deltaG);
			} else {
				scale = initialScale;
			}

			// Scale initial matrix with ``scale * np.eye(n)``
			switch (approxType) {
			case HESSIAN:
				B = B.times(scale);
				break;
			case INV_HESSIAN:
				H = H.times(scale);
				break;
			default:
				throw new RuntimeException("not implemented");
			}

			firstIteration = false;
		}

		updateImplementation(deltaX, deltaG);
	}

	@Override
	public Matrix dot(Matrix p) {

		// TODO: use _symv from LAPACK

		switch (approxType) {
		case HESSIAN:
			return B.mtimes(p);
		case INV_HESSIAN:
			return H.mtimes(p);
		default:
			throw new RuntimeException("not implemented");
		}
	}

	@Override
	public Matrix getMatrix() {
		switch (approxType) {
		case HESSIAN:
			return DenseMatrix.Factory.copyFromMatrix(B);
		case INV_HESSIAN:
			return DenseMatrix.Factory.copyFromMatrix(H);
		default:
			throw new RuntimeException("not implemented");
		}
	}
}
