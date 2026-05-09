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
package de.labathome.trustconstr;

import java.util.ArrayList;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.EqIneqSplit;
import de.labathome.trustconstr.records.PreparedConstraint;

/**
 * Canonical-form constraint
 *
 * <pre>
 *     f_eq(x)   = 0
 *     f_ineq(x) &le; 0
 * </pre>
 *
 * Mirrors scipy's {@code CanonicalConstraint}
 * ({@code scipy/optimize/_trustregion_constr/canonical_constraint.py}).
 *
 * <p>The Java port already canonicalises bounds-form constraints inline inside
 * {@link LinearConstraint} and {@link NonlinearConstraint} (see those classes'
 * 4-block ineq ordering: {@code less | greater | interval-upper |
 * interval-lower}). This class wraps that machinery into the scipy-shape API
 * for parity with {@code test_canonical_constraint.py}.
 *
 * <p>The class itself is immutable; all factory methods produce a new
 * instance carrying lambdas that delegate to the underlying constraint
 * objects.
 */
public final class CanonicalConstraint {

	/** Number of canonical equality rows. */
	public final int nEq;

	/** Number of canonical inequality rows. */
	public final int nIneq;

	/**
	 * Per-canonical-inequality-row strict-feasibility flags (length
	 * {@link #nIneq}). Mirrors scipy's {@code keep_feasible} attribute.
	 */
	public final boolean[] keepFeasible;

	private final Function<Matrix, EqIneqSplit> funImpl;
	private final Function<Matrix, EqIneqSplit> jacImpl;
	private final HessianFn hessImpl;

	/**
	 * Functional interface for the constraint Hessian-of-Lagrangian:
	 * {@code Sum_i v_eq[i] * H_{eq,i}(x) + Sum_j v_ineq[j] * H_{ineq,j}(x)}.
	 */
	@FunctionalInterface
	public interface HessianFn {
		/**
		 * @param x     current iterate ({@code n x 1})
		 * @param vEq   equality multipliers, length {@code nEq}; may be
		 *              {@code null} when {@code nEq == 0}
		 * @param vIneq inequality multipliers, length {@code nIneq}; may be
		 *              {@code null} when {@code nIneq == 0}
		 * @return summed canonical-constraint Hessian ({@code n x n})
		 */
		Matrix apply(Matrix x, double[] vEq, double[] vIneq);
	}

	private CanonicalConstraint(int nEq, int nIneq, boolean[] keepFeasible,
			Function<Matrix, EqIneqSplit> funImpl,
			Function<Matrix, EqIneqSplit> jacImpl,
			HessianFn hessImpl) {
		this.nEq = nEq;
		this.nIneq = nIneq;
		this.keepFeasible = keepFeasible;
		this.funImpl = funImpl;
		this.jacImpl = jacImpl;
		this.hessImpl = hessImpl;
	}

	/**
	 * Evaluate the canonical equality and inequality residuals at {@code x}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return {@code (cEq, cIneq)} where {@code cEq} is {@code nEq x 1} and
	 *         {@code cIneq} is {@code nIneq x 1}
	 */
	public EqIneqSplit fun(Matrix x) {
		return funImpl.apply(x);
	}

	/**
	 * Evaluate the canonical equality and inequality Jacobians at {@code x}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return {@code (JEq, JIneq)} where {@code JEq} is {@code nEq x n} and
	 *         {@code JIneq} is {@code nIneq x n}
	 */
	public EqIneqSplit jac(Matrix x) {
		return jacImpl.apply(x);
	}

	/**
	 * Evaluate the constraint Hessian-of-Lagrangian at {@code x}.
	 *
	 * @param x     current iterate ({@code n x 1})
	 * @param vEq   equality multipliers, length {@link #nEq}; may be
	 *              {@code null} when {@link #nEq} is zero
	 * @param vIneq inequality multipliers, length {@link #nIneq}; may be
	 *              {@code null} when {@link #nIneq} is zero
	 * @return summed Hessian contribution ({@code n x n}); zeros for linear
	 *         and empty constraints
	 */
	public Matrix hess(Matrix x, double[] vEq, double[] vIneq) {
		return hessImpl.apply(x, vEq, vIneq);
	}

	/**
	 * Construct an empty canonical constraint -- zero rows of each kind. The
	 * Jacobians are returned as {@code 0 x n} matrices, the Hessian as an
	 * {@code n x n} zero matrix. Mirrors scipy's
	 * {@code CanonicalConstraint.empty(n)}.
	 *
	 * @param n number of decision variables
	 * @return an empty canonical constraint sized for {@code n} variables
	 */
	public static CanonicalConstraint empty(int n) {
		Function<Matrix, EqIneqSplit> funImpl = x -> new EqIneqSplit(
				Matrix.Factory.zeros(0, 1), Matrix.Factory.zeros(0, 1));
		Function<Matrix, EqIneqSplit> jacImpl = x -> new EqIneqSplit(
				Matrix.Factory.zeros(0, n), Matrix.Factory.zeros(0, n));
		HessianFn hessImpl = (x, vEq, vIneq) -> Matrix.Factory.zeros(n, n);
		return new CanonicalConstraint(0, 0, new boolean[0], funImpl, jacImpl, hessImpl);
	}

	/**
	 * Wrap a {@link LinearConstraint} as a canonical constraint. The underlying
	 * row classification (equal / less / greater / interval-upper /
	 * interval-lower) is already done by {@code LinearConstraint}; this factory
	 * just exposes its {@code constrEq}/{@code constrIneq}/{@code jacEq}/
	 * {@code jacIneq} via the scipy-shape API.
	 *
	 * <p>Linear constraints have zero constraint Hessian, so {@code hess(...)}
	 * returns an {@code n x n} zero matrix.
	 *
	 * @param lc underlying linear constraint
	 * @return canonical wrapper over {@code lc}
	 */
	public static CanonicalConstraint fromLinearConstraint(LinearConstraint lc) {
		int nEq = lc.nEq();
		int nIneq = lc.nIneq();
		boolean[] kf = lc.enforceFeasibilityIneq();
		long n = lc.getA().getColumnCount();
		Function<Matrix, EqIneqSplit> funImpl = x -> new EqIneqSplit(
				lc.constrEq(x), lc.constrIneq(x));
		Function<Matrix, EqIneqSplit> jacImpl = x -> new EqIneqSplit(
				lc.jacEq(x), lc.jacIneq(x));
		HessianFn hessImpl = (x, vEq, vIneq) -> Matrix.Factory.zeros(n, n);
		return new CanonicalConstraint(nEq, nIneq, kf, funImpl, jacImpl, hessImpl);
	}

	/**
	 * Wrap a {@link NonlinearConstraint} as a canonical constraint. Mirrors the
	 * dispatch done by scipy's {@code from_PreparedConstraint} on a constraint
	 * that came from a user-supplied {@code NonlinearConstraint}.
	 *
	 * <p>If the underlying constraint was constructed without an analytic
	 * {@code hess}, {@code hess(x, vEq, vIneq)} returns an {@code n x n} zero
	 * matrix (no Hessian contribution).
	 *
	 * @param nc underlying nonlinear constraint
	 * @param n  number of decision variables (cannot be inferred from
	 *           {@code NonlinearConstraint}, which only knows the
	 *           constraint-output dimension {@code m})
	 * @return canonical wrapper over {@code nc}
	 */
	public static CanonicalConstraint fromNonlinearConstraint(NonlinearConstraint nc, int n) {
		int nEq = nc.nEq();
		int nIneq = nc.nIneq();
		boolean[] kf = nc.enforceFeasibilityIneq();
		Function<Matrix, EqIneqSplit> funImpl = x -> new EqIneqSplit(
				nc.constrEq(x), nc.constrIneq(x));
		Function<Matrix, EqIneqSplit> jacImpl = x -> new EqIneqSplit(
				nc.jacEq(x), nc.jacIneq(x));
		HessianFn hessImpl = (x, vEq, vIneq) -> {
			Matrix part = nc.lagrangianContribution(x, vEq, vIneq);
			return (part != null) ? part : Matrix.Factory.zeros(n, n);
		};
		return new CanonicalConstraint(nEq, nIneq, kf, funImpl, jacImpl, hessImpl);
	}

	/**
	 * Promote {@link Bounds} into a canonical constraint via
	 * {@link LinearConstraint#fromBounds(Bounds)}. Mirrors scipy's
	 * {@code PreparedConstraint(bounds, ...)} -> {@code from_PreparedConstraint}
	 * shortcut.
	 *
	 * @param bounds box bounds on the decision variables
	 * @return canonical constraint enforcing {@code lb &le; x &le; ub}
	 */
	public static CanonicalConstraint fromBounds(Bounds bounds) {
		return fromLinearConstraint(LinearConstraint.fromBounds(bounds));
	}

	/**
	 * Build a canonical constraint from a {@link PreparedConstraint}. Mirrors
	 * scipy's {@code CanonicalConstraint.from_PreparedConstraint}: dispatches
	 * on the original source constraint type carried by {@link
	 * PreparedConstraint#source()}.
	 *
	 * @param pc prepared constraint with cached {@code f}, {@code J} at x0
	 * @return canonical wrapper over the source constraint
	 * @throws IllegalArgumentException if the source is not one of
	 *         {@link LinearConstraint}, {@link NonlinearConstraint}, or
	 *         {@link Bounds}
	 */
	public static CanonicalConstraint fromPreparedConstraint(PreparedConstraint pc) {
		Object src = pc.source();
		if (src instanceof LinearConstraint lc) {
			return fromLinearConstraint(lc);
		}
		if (src instanceof NonlinearConstraint nc) {
			return fromNonlinearConstraint(nc, (int) pc.n());
		}
		if (src instanceof Bounds b) {
			return fromBounds(b);
		}
		throw new IllegalArgumentException(
				"Unsupported PreparedConstraint source type: "
						+ (src == null ? "null" : src.getClass().getName()));
	}

	/**
	 * Concatenate a list of canonical constraints into one. Equality blocks
	 * are stacked first across all sources, then inequality blocks across all
	 * sources. Mirrors scipy's
	 * {@code CanonicalConstraint.concatenate(canonical_constraints, sparse_jacobian)}.
	 *
	 * <p>The resulting Jacobian's storage layout follows {@code sparseJacobian}:
	 * {@code true} produces a {@link SparseMatrix}, {@code false} produces a
	 * dense {@link Matrix}. The summed Hessian is the (matrix) sum of each
	 * source's Hessian applied to its slice of the multipliers.
	 *
	 * @param sources         canonical constraints to concatenate; row order
	 *                        follows the list order
	 * @param sparseJacobian  if {@code true}, the combined Jacobian is built as
	 *                        a {@link SparseMatrix}; otherwise dense
	 * @return concatenated canonical constraint
	 */
	public static CanonicalConstraint concatenate(List<CanonicalConstraint> sources,
			boolean sparseJacobian) {
		int totalEq = 0;
		int totalIneq = 0;
		for (CanonicalConstraint c : sources) {
			totalEq += c.nEq;
			totalIneq += c.nIneq;
		}
		final int finalEq = totalEq;
		final int finalIneq = totalIneq;
		boolean[] kf = new boolean[totalIneq];
		int kfPos = 0;
		for (CanonicalConstraint c : sources) {
			System.arraycopy(c.keepFeasible, 0, kf, kfPos, c.nIneq);
			kfPos += c.nIneq;
		}

		Function<Matrix, EqIneqSplit> funImpl = x -> {
			Matrix combinedEq = Matrix.Factory.zeros(finalEq, 1);
			Matrix combinedIneq = Matrix.Factory.zeros(finalIneq, 1);
			int eqRow = 0;
			int ineqRow = 0;
			for (CanonicalConstraint c : sources) {
				EqIneqSplit s = c.fun(x);
				for (int i = 0; i < c.nEq; ++i) {
					combinedEq.setAsDouble(s.eq().getAsDouble(i, 0), eqRow + i, 0);
				}
				for (int i = 0; i < c.nIneq; ++i) {
					combinedIneq.setAsDouble(s.ineq().getAsDouble(i, 0), ineqRow + i, 0);
				}
				eqRow += c.nEq;
				ineqRow += c.nIneq;
			}
			return new EqIneqSplit(combinedEq, combinedIneq);
		};

		Function<Matrix, EqIneqSplit> jacImpl = x -> {
			long n = x.getRowCount();
			Matrix combinedEq = sparseJacobian
					? SparseMatrix.Factory.zeros(finalEq, n)
					: Matrix.Factory.zeros(finalEq, n);
			Matrix combinedIneq = sparseJacobian
					? SparseMatrix.Factory.zeros(finalIneq, n)
					: Matrix.Factory.zeros(finalIneq, n);
			int eqRow = 0;
			int ineqRow = 0;
			for (CanonicalConstraint c : sources) {
				EqIneqSplit s = c.jac(x);
				copyBlock(s.eq(), combinedEq, eqRow, c.nEq, n, sparseJacobian);
				copyBlock(s.ineq(), combinedIneq, ineqRow, c.nIneq, n, sparseJacobian);
				eqRow += c.nEq;
				ineqRow += c.nIneq;
			}
			return new EqIneqSplit(combinedEq, combinedIneq);
		};

		HessianFn hessImpl = (x, vEq, vIneq) -> {
			Matrix total = null;
			int eqOff = 0;
			int ineqOff = 0;
			for (CanonicalConstraint c : sources) {
				double[] vEqSlice = (vEq != null && c.nEq > 0)
						? Arrays.copyOfRange(vEq, eqOff, eqOff + c.nEq)
						: null;
				double[] vIneqSlice = (vIneq != null && c.nIneq > 0)
						? Arrays.copyOfRange(vIneq, ineqOff, ineqOff + c.nIneq)
						: null;
				Matrix part = c.hess(x, vEqSlice, vIneqSlice);
				if (part != null) {
					total = (total == null) ? part : total.plus(part);
				}
				eqOff += c.nEq;
				ineqOff += c.nIneq;
			}
			if (total == null) {
				long n = x.getRowCount();
				return Matrix.Factory.zeros(n, n);
			}
			return total;
		};

		return new CanonicalConstraint(totalEq, totalIneq, kf, funImpl, jacImpl, hessImpl);
	}

	private static void copyBlock(Matrix src, Matrix dst, int dstRowOffset,
			int rows, long cols, boolean sparseDst) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double v = src.getAsDouble(i, j);
				if (v != 0.0 || !sparseDst) {
					dst.setAsDouble(v, dstRowOffset + i, j);
				}
			}
		}
	}

	/**
	 * Output shape of {@link #initialConstraintsAsCanonical}: the four arrays
	 * {@code (cEq, cIneq, JEq, JIneq)} returned as a record. Mirrors scipy's
	 * 4-tuple return value.
	 *
	 * @param cEq   equality residual at {@code x0} ({@code nEq x 1})
	 * @param cIneq inequality residual at {@code x0} ({@code nIneq x 1})
	 * @param JEq   equality Jacobian at {@code x0} ({@code nEq x n})
	 * @param JIneq inequality Jacobian at {@code x0} ({@code nIneq x n})
	 */
	public record InitialCanonical(Matrix cEq, Matrix cIneq, Matrix JEq, Matrix JIneq) {}

	/**
	 * Stack the canonical residuals and Jacobians of every constraint in
	 * {@code canonicals} at the starting iterate {@code x0}. Mirrors scipy's
	 * top-level
	 * {@code initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian)}
	 * function.
	 *
	 * <p>scipy avoids re-evaluating each constraint at {@code x0} by reusing
	 * the cached {@code f}, {@code J} stored in {@code PreparedConstraint};
	 * the Java port currently re-evaluates because {@link
	 * de.labathome.trustconstr.records.PreparedConstraint} does not yet
	 * cache (Phase 2). Numerically the result is identical.
	 *
	 * @param n              number of decision variables (used to size the
	 *                       empty Jacobians when {@code canonicals} is empty)
	 * @param canonicals     canonical constraints to evaluate and stack
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian if {@code true}, the combined Jacobian is built as
	 *                       a {@link SparseMatrix}; otherwise dense
	 * @return stacked residuals and Jacobians
	 */
	public static InitialCanonical initialConstraintsAsCanonical(int n,
			List<CanonicalConstraint> canonicals, Matrix x0, boolean sparseJacobian) {
		if (canonicals.isEmpty()) {
			Matrix cEq = Matrix.Factory.zeros(0, 1);
			Matrix cIneq = Matrix.Factory.zeros(0, 1);
			Matrix JEq = sparseJacobian
					? SparseMatrix.Factory.zeros(0, n)
					: Matrix.Factory.zeros(0, n);
			Matrix JIneq = sparseJacobian
					? SparseMatrix.Factory.zeros(0, n)
					: Matrix.Factory.zeros(0, n);
			return new InitialCanonical(cEq, cIneq, JEq, JIneq);
		}
		CanonicalConstraint combined = concatenate(canonicals, sparseJacobian);
		EqIneqSplit fun = combined.fun(x0);
		EqIneqSplit jac = combined.jac(x0);
		return new InitialCanonical(fun.eq(), fun.ineq(), jac.eq(), jac.ineq());
	}

	/**
	 * scipy-shape overload taking a list of {@link PreparedConstraint}s.
	 * Mirrors scipy's
	 * {@code initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian)}.
	 *
	 * <p>Each prepared constraint is converted to a {@link CanonicalConstraint}
	 * via {@link #fromPreparedConstraint}, and the resulting canonicals are
	 * stacked. The cached iterate {@code x0} from the first prepared
	 * constraint is used to evaluate the stacked residuals and Jacobians.
	 *
	 * @param n              number of decision variables
	 * @param prepared       prepared constraints
	 * @param sparseJacobian if {@code true}, the combined Jacobian is built
	 *                       sparse; otherwise dense
	 * @return stacked residuals and Jacobians at {@code x0}
	 */
	public static InitialCanonical initialConstraintsAsCanonical(int n,
			List<PreparedConstraint> prepared, boolean sparseJacobian) {
		if (prepared.isEmpty()) {
			Matrix cEq = Matrix.Factory.zeros(0, 1);
			Matrix cIneq = Matrix.Factory.zeros(0, 1);
			Matrix JEq = sparseJacobian
					? SparseMatrix.Factory.zeros(0, n)
					: Matrix.Factory.zeros(0, n);
			Matrix JIneq = sparseJacobian
					? SparseMatrix.Factory.zeros(0, n)
					: Matrix.Factory.zeros(0, n);
			return new InitialCanonical(cEq, cIneq, JEq, JIneq);
		}
		Matrix x0 = prepared.get(0).x0();
		List<CanonicalConstraint> canonicals = new ArrayList<>(prepared.size());
		for (PreparedConstraint pc : prepared) {
			canonicals.add(fromPreparedConstraint(pc));
		}
		return initialConstraintsAsCanonical(n, canonicals, x0, sparseJacobian);
	}
}
