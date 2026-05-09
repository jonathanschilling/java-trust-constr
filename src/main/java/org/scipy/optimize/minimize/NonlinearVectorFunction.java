package org.scipy.optimize.minimize;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.scipy.optimize.minimize.interfaces.VectorFunctionLike;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;

/**
 * Light-weight nonlinear vector function carrier for use inside
 * {@link org.scipy.optimize.minimize.records.PreparedConstraint}.
 *
 * <p>Wraps a user-supplied {@code fun(x)}, {@code jac(x)}, optional
 * {@code hess(x, v)} triple, caching {@code f} and {@code J} at the iterate
 * passed to the constructor. Subsequent {@link #fun(Matrix)} /
 * {@link #jac(Matrix)} calls re-evaluate when {@code x} changes.
 *
 * <p>This class is the analytic-derivatives counterpart of scipy's
 * {@code VectorFunction} when {@code jac} and {@code hess} are both supplied
 * by the caller. The full {@link org.scipy.optimize.minimize.VectorFunction}
 * carries the finite-difference / quasi-Newton fallbacks; this class skips
 * them. {@link org.scipy.optimize.minimize.NonlinearConstraint} already
 * resolves an FD Jacobian when the user omits one (its constructor wraps
 * a 2-point {@code NumDiff} closure), so by the time we reach this class
 * {@code jac} is always a working callable.
 */
public class NonlinearVectorFunction implements VectorFunctionLike {

	private final Function<Matrix, Matrix> userFun;
	private final Function<Matrix, Matrix> userJac;
	private final BiFunction<Matrix, Matrix, Matrix> userHess;

	private final boolean sparseJacobian;
	private final long m;
	private final long n;

	private Matrix x;
	private Matrix f;
	private Matrix J;
	private Matrix v;
	private boolean fUpdated;
	private boolean jUpdated;
	private int numFunctionEvals;
	private int numJacobianEvals;
	private int numHessianEvals;

	/**
	 * Build a nonlinear vector function and cache {@code f}, {@code J} at
	 * {@code x0}.
	 *
	 * @param fun            user constraint function {@code R^n -> R^m}
	 * @param jac            analytic Jacobian (or pre-built FD closure)
	 *                       {@code R^n -> R^{m x n}}
	 * @param hess           optional constraint Hessian-of-Lagrangian
	 *                       {@code (x, v) -> Sum v[i] H_{f_i}(x)}; {@code null}
	 *                       is permitted (then {@link #hess(Matrix, Matrix)}
	 *                       returns zeros)
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian {@code Optional.of(true)} forces sparse Jacobian
	 *                       storage; {@code Optional.of(false)} forces dense;
	 *                       {@code Optional.empty()} preserves whatever
	 *                       {@code jac.apply(x0)} returns
	 */
	public NonlinearVectorFunction(Function<Matrix, Matrix> fun,
			Function<Matrix, Matrix> jac,
			BiFunction<Matrix, Matrix, Matrix> hess,
			Matrix x0, Optional<Boolean> sparseJacobian) {
		this.userFun = fun;
		this.userJac = jac;
		this.userHess = hess;
		this.numFunctionEvals = 0;
		this.numJacobianEvals = 0;
		this.numHessianEvals = 0;
		this.x = Matrix.Factory.copyFromMatrix(x0);
		this.f = userFun.apply(this.x);
		this.numFunctionEvals++;
		Matrix rawJ = userJac.apply(this.x);
		this.numJacobianEvals++;
		boolean wantSparse;
		if (sparseJacobian.isPresent()) {
			wantSparse = sparseJacobian.get();
		} else {
			wantSparse = rawJ.isSparse();
		}
		if (wantSparse) {
			this.J = rawJ.isSparse() ? rawJ : SparseMatrix.Factory.copyFromMatrix(rawJ);
			this.sparseJacobian = true;
		} else {
			this.J = rawJ.isSparse() ? DenseMatrix.Factory.copyFromMatrix(rawJ) : rawJ;
			this.sparseJacobian = false;
		}
		this.fUpdated = true;
		this.jUpdated = true;
		this.m = this.f.getRowCount();
		this.n = x0.getRowCount();
		this.v = Matrix.Factory.zeros(this.m, 1);
	}

	private void updateX(Matrix newX) {
		if (!matrixEquals(newX, x)) {
			x = Matrix.Factory.copyFromMatrix(newX);
			fUpdated = false;
			jUpdated = false;
		}
	}

	@Override
	public Matrix fun(Matrix newX) {
		updateX(newX);
		if (!fUpdated) {
			f = userFun.apply(x);
			numFunctionEvals++;
			fUpdated = true;
		}
		return f;
	}

	@Override
	public Matrix jac(Matrix newX) {
		updateX(newX);
		if (!jUpdated) {
			Matrix rawJ = userJac.apply(x);
			numJacobianEvals++;
			if (sparseJacobian) {
				J = rawJ.isSparse() ? rawJ : SparseMatrix.Factory.copyFromMatrix(rawJ);
			} else {
				J = rawJ.isSparse() ? DenseMatrix.Factory.copyFromMatrix(rawJ) : rawJ;
			}
			jUpdated = true;
		}
		return J;
	}

	@Override
	public Matrix hess(Matrix newX, Matrix newV) {
		updateX(newX);
		this.v = newV;
		if (userHess == null) {
			return Matrix.Factory.zeros(n, n);
		}
		numHessianEvals++;
		return userHess.apply(x, newV);
	}

	@Override
	public Matrix v() {
		return v;
	}

	@Override
	public int numFunctionEvals() {
		return numFunctionEvals;
	}

	@Override
	public int numJacobianEvals() {
		return numJacobianEvals;
	}

	@Override
	public int numHessianEvals() {
		return numHessianEvals;
	}

	@Override
	public Matrix f() {
		return f;
	}

	@Override
	public Matrix J() {
		return J;
	}

	@Override
	public long m() {
		return m;
	}

	@Override
	public long n() {
		return n;
	}

	/** @return {@code true} if {@link #J} is stored as a {@link SparseMatrix} */
	public boolean sparseJacobian() {
		return sparseJacobian;
	}

	private static boolean matrixEquals(Matrix a, Matrix b) {
		if (a.getRowCount() != b.getRowCount()
				|| a.getColumnCount() != b.getColumnCount()) {
			return false;
		}
		for (long i = 0; i < a.getRowCount(); ++i) {
			for (long j = 0; j < a.getColumnCount(); ++j) {
				if (a.getAsDouble(i, j) != b.getAsDouble(i, j)) {
					return false;
				}
			}
		}
		return true;
	}
}
