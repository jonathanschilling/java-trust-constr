package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

/** Java port of scipy.optimize.minimize(method='trust-constr') */
public class MinimizeTrustConstr {

	public static State updateState(State state, Matrix x,
			boolean lastIterationFailed, ScalarFunction objective,
			PreparedConstraint[] preparedConstraints, long startTime,
			double trustRadius, double constraintPenalty, CGInfo cgInfo) {

		state.nIter++;

		state.numEval = objective.numEval();
		state.numGradientEval = objective.numGradientEval();
		state.numHessianEval = objective.numHessianEval();

		int n = preparedConstraints.length;
		if (state.numConstraintEval == null || state.numConstraintEval.length != n) {
			state.numConstraintEval = new int[n];
			state.numConstraintJacobianEval = new int[n];
			state.numConstraintHessianEval = new int[n];

			state.v = new Matrix[n];
			state.constr = new Matrix[n];
			state.jac = new Matrix[n];
		}
		for (int i=0; i<preparedConstraints.length; ++i) {
			VectorFunction c = preparedConstraints[i].fun;
			state.numConstraintEval[i] = c.numEval();
			state.numConstraintJacobianEval[i] = c.numGradientEval();
			state.numConstraintHessianEval[i] = c.numHessianEval();
		}

		if (!lastIterationFailed) {
			state.x = x;
			state.fun = objective.lastEval();
			state.grad = objective.lastGradient();

			for (int i=0; i<preparedConstraints.length; ++i) {
				VectorFunction c = preparedConstraints[i].fun;
				state.v[i] = c.lastV();
				state.constr[i] = c.lastEval();
				state.jac[i] = c.lastGradient();
			}

			// Compute Lagrangian Gradient
			state.lagrangianGrad = Matrix.Factory.copyFromMatrix(state.grad);
			for (PreparedConstraint c: preparedConstraints) {
				state.lagrangianGrad = state.lagrangianGrad.plus(c.fun.lastGradient().transpose().mtimes(c.fun.lastV()));
			}
			state.optimality = state.lagrangianGrad.normInf();

			// Compute maximum constraint violation
			state.constrViolation = 0.0;
			for (int i=0; i<preparedConstraints.length; ++i) {
				double[] lb = preparedConstraints[i].lb;
				double[] ub = preparedConstraints[i].ub;
				Matrix c= state.constr[i];
				for (int j=0; j<lb.length; ++j) {
					double lowerViolation = lb[j] - c.getAsDouble(j, 0);
					double upperViolation = c.getAsDouble(j, 0) - ub[j];
					double maxCViol = Math.max(lowerViolation, upperViolation);
					state.constrViolation = Math.max(state.constrViolation, maxCViol);
				}
			}
		}

		state.executionTime = System.nanoTime() - startTime;

		state.trustRadius = trustRadius;
		state.constraintPenalty = constraintPenalty;

		state.cgNIter += cgInfo.niter;
		state.cgStopCond = cgInfo.stopCond;

		return state;
	}

	public static void minimizeTrustConstr() {






	}

}
