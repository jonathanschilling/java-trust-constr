package org.scipy.optimize.minimize;

import java.util.List;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.ujmp.core.Matrix;

/**
 * Concatenation of several {@link LinearConstraint} / {@link NonlinearConstraint}
 * objects into a single {@link Constraint} + {@link Jacobian} that the
 * orchestrator can pass straight to
 * {@link EqualityConstrainedSQP#eqSQP} or
 * {@link TrustRegionInteriorPoint#trustRegionInteriorPoint}.
 *
 * <p>The combined constraint stacks all equality rows from all sources, then
 * all inequality rows from all sources. This mirrors scipy's
 * {@code CanonicalConstraint.concatenate} on a smaller scale (without the
 * Hessian-of-Lagrangian aggregation, which the orchestrator currently doesn't
 * use for nonlinear constraints).
 *
 * <p>Each input must implement both {@link Constraint} and {@link Jacobian}
 * and expose its {@code nEq()} / {@code nIneq()} via the value-class types
 * {@link LinearConstraint} or {@link NonlinearConstraint}.
 */
public final class CombinedConstraint implements Constraint, Jacobian {

	private final Object[] sources;
	private final int nVars;
	private final int totalEq;
	private final int totalIneq;

	public <C> CombinedConstraint(List<C> sources, int nVars) {
		this(sources.toArray(), nVars);
	}

	public CombinedConstraint(Object[] sources, int nVars) {
		this.sources = sources.clone();
		this.nVars = nVars;
		int e = 0;
		int i = 0;
		for (Object s : sources) {
			if (s instanceof LinearConstraint) {
				e += ((LinearConstraint) s).nEq();
				i += ((LinearConstraint) s).nIneq();
			} else if (s instanceof NonlinearConstraint) {
				e += ((NonlinearConstraint) s).nEq();
				i += ((NonlinearConstraint) s).nIneq();
			} else {
				throw new IllegalArgumentException("Unsupported constraint type: "
						+ (s == null ? "null" : s.getClass().getName()));
			}
		}
		this.totalEq = e;
		this.totalIneq = i;
	}

	public int nEq() { return totalEq; }
	public int nIneq() { return totalIneq; }

	@Override
	public Matrix constrEq(Matrix x) {
		if (totalEq == 0) {
			return Matrix.Factory.zeros(0, 1);
		}
		Matrix out = Matrix.Factory.zeros(totalEq, 1);
		int row = 0;
		for (Object s : sources) {
			Constraint c = (Constraint) s;
			int nEq = (s instanceof LinearConstraint)
					? ((LinearConstraint) s).nEq()
					: ((NonlinearConstraint) s).nEq();
			if (nEq == 0) continue;
			Matrix part = c.constrEq(x);
			for (int k = 0; k < nEq; ++k) {
				out.setAsDouble(part.getAsDouble(k, 0), row + k, 0);
			}
			row += nEq;
		}
		return out;
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		if (totalIneq == 0) {
			return Matrix.Factory.zeros(0, 1);
		}
		Matrix out = Matrix.Factory.zeros(totalIneq, 1);
		int row = 0;
		for (Object s : sources) {
			Constraint c = (Constraint) s;
			int nIneq = (s instanceof LinearConstraint)
					? ((LinearConstraint) s).nIneq()
					: ((NonlinearConstraint) s).nIneq();
			if (nIneq == 0) continue;
			Matrix part = c.constrIneq(x);
			for (int k = 0; k < nIneq; ++k) {
				out.setAsDouble(part.getAsDouble(k, 0), row + k, 0);
			}
			row += nIneq;
		}
		return out;
	}

	@Override
	public Matrix jacEq(Matrix x) {
		if (totalEq == 0) {
			return Matrix.Factory.zeros(0, nVars);
		}
		Matrix out = Matrix.Factory.zeros(totalEq, nVars);
		int row = 0;
		for (Object s : sources) {
			Jacobian j = (Jacobian) s;
			int nEq = (s instanceof LinearConstraint)
					? ((LinearConstraint) s).nEq()
					: ((NonlinearConstraint) s).nEq();
			if (nEq == 0) continue;
			Matrix part = j.jacEq(x);
			for (int k = 0; k < nEq; ++k) {
				for (int c = 0; c < nVars; ++c) {
					out.setAsDouble(part.getAsDouble(k, c), row + k, c);
				}
			}
			row += nEq;
		}
		return out;
	}

	/**
	 * Sum of constraint-Hessian-of-Lagrangian contributions across all sources.
	 * Walks the constraint list, slices {@code vEq}/{@code vIneq} into the
	 * portions that belong to each source (in declaration order), and sums the
	 * resulting Hessians. Returns {@code null} if no source contributes
	 * (e.g. all sources are {@link LinearConstraint}, which has zero Hessian).
	 */
	public Matrix lagrangianContribution(Matrix x, double[] vEq, double[] vIneq) {
		Matrix total = null;
		int eqOffset = 0;
		int ineqOffset = 0;
		for (Object s : sources) {
			int srcEq;
			int srcIneq;
			Matrix part = null;
			if (s instanceof LinearConstraint) {
				srcEq = ((LinearConstraint) s).nEq();
				srcIneq = ((LinearConstraint) s).nIneq();
				// Linear constraint Hessian is identically 0; skip.
			} else if (s instanceof NonlinearConstraint) {
				NonlinearConstraint nc = (NonlinearConstraint) s;
				srcEq = nc.nEq();
				srcIneq = nc.nIneq();
				double[] vEqSlice = new double[srcEq];
				double[] vIneqSlice = new double[srcIneq];
				System.arraycopy(vEq, eqOffset, vEqSlice, 0, srcEq);
				System.arraycopy(vIneq, ineqOffset, vIneqSlice, 0, srcIneq);
				part = nc.lagrangianContribution(x, vEqSlice, vIneqSlice);
			} else {
				throw new IllegalStateException("Unsupported constraint type: "
						+ s.getClass().getName());
			}
			if (part != null) {
				total = (total == null) ? part : total.plus(part);
			}
			eqOffset += srcEq;
			ineqOffset += srcIneq;
		}
		return total;
	}

	@Override
	public Matrix jacIneq(Matrix x) {
		if (totalIneq == 0) {
			return Matrix.Factory.zeros(0, nVars);
		}
		Matrix out = Matrix.Factory.zeros(totalIneq, nVars);
		int row = 0;
		for (Object s : sources) {
			Jacobian j = (Jacobian) s;
			int nIneq = (s instanceof LinearConstraint)
					? ((LinearConstraint) s).nIneq()
					: ((NonlinearConstraint) s).nIneq();
			if (nIneq == 0) continue;
			Matrix part = j.jacIneq(x);
			for (int k = 0; k < nIneq; ++k) {
				for (int c = 0; c < nVars; ++c) {
					out.setAsDouble(part.getAsDouble(k, c), row + k, c);
				}
			}
			row += nIneq;
		}
		return out;
	}
}
