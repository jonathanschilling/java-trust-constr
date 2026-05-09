package de.labathome.trustconstr;

import java.util.Optional;

import de.labathome.trustconstr.interfaces.VectorFunctionLike;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

/**
 * Linear vector function {@code f(x) = A x} and its derivatives.
 *
 * <p>Counterpart to scipy's {@code LinearVectorFunction}
 * ({@code scipy/optimize/_differentiable_functions.py:769}). The Jacobian is
 * the constant matrix {@code A}; the Hessian-of-Lagrangian is identically
 * zero (returned as an {@code n x n} sparse zero matrix).
 *
 * <p>Caches the most recently evaluated point and {@code f} so repeated
 * {@link #fun(Matrix)} calls at the same {@code x} cost a single matvec.
 */
public class LinearVectorFunction implements VectorFunctionLike {

	/** Constraint matrix; mirrors scipy's {@code self.J}. */
	protected final Matrix A;

	/** Output dimension ({@code A.rows}). */
	protected final long m;

	/** Input dimension ({@code A.cols}). */
	protected final long n;

	/** {@code true} if {@code A} is stored as a {@link SparseMatrix}. */
	protected final boolean sparseJacobian;

	/** Constant zero Hessian ({@code n x n}); shared across calls. */
	protected final Matrix H;

	/** Most recently evaluated point. */
	protected Matrix x;

	/** Cached {@code A x} at {@link #x}; {@code null} until first call. */
	protected Matrix f;

	/** {@code true} when {@link #f} matches the current {@link #x}. */
	protected boolean fUpdated;

	/**
	 * Most recently set Lagrange multipliers (from {@link #hess(Matrix, Matrix)}).
	 * Stored only to mirror scipy's API; not used for any computation.
	 */
	protected Matrix v;

	/**
	 * Build a linear vector function {@code f(x) = A x}.
	 *
	 * @param A              constraint matrix ({@code m x n})
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian {@code Optional.of(true)} forces sparse storage,
	 *                       {@code Optional.of(false)} forces dense,
	 *                       {@code Optional.empty()} preserves whatever
	 *                       {@code A} already is
	 */
	public LinearVectorFunction(Matrix A, Matrix x0, Optional<Boolean> sparseJacobian) {
		boolean wantSparse;
		if (sparseJacobian.isPresent()) {
			wantSparse = sparseJacobian.get();
		} else {
			wantSparse = A.isSparse();
		}
		if (wantSparse) {
			this.A = A.isSparse() ? A : SparseMatrix.Factory.copyFromMatrix(A);
			this.sparseJacobian = true;
		} else {
			this.A = A.isSparse() ? DenseMatrix.Factory.copyFromMatrix(A) : A;
			this.sparseJacobian = false;
		}
		this.m = this.A.getRowCount();
		this.n = this.A.getColumnCount();
		this.x = Matrix.Factory.copyFromMatrix(x0);
		this.f = this.A.mtimes(this.x);
		this.fUpdated = true;
		this.H = SparseMatrix.Factory.zeros(this.n, this.n);
		this.v = Matrix.Factory.zeros(this.m, 1);
	}

	private void updateX(Matrix newX) {
		if (!matrixEquals(newX, x)) {
			x = Matrix.Factory.copyFromMatrix(newX);
			fUpdated = false;
		}
	}

	@Override
	public Matrix fun(Matrix newX) {
		updateX(newX);
		if (!fUpdated) {
			f = A.mtimes(x);
			fUpdated = true;
		}
		return f;
	}

	@Override
	public Matrix jac(Matrix newX) {
		updateX(newX);
		return A;
	}

	@Override
	public Matrix hess(Matrix newX, Matrix newV) {
		updateX(newX);
		this.v = newV;
		return H;
	}

	@Override
	public Matrix f() {
		return f;
	}

	@Override
	public Matrix J() {
		return A;
	}

	@Override
	public long m() {
		return m;
	}

	@Override
	public long n() {
		return n;
	}

	/** @return {@code true} if the Jacobian is stored as a {@link SparseMatrix} */
	public boolean sparseJacobian() {
		return sparseJacobian;
	}

	@Override
	public Matrix v() {
		return v;
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
