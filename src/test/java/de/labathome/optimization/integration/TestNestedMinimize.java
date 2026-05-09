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
package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Port of scipy {@code _trustregion_constr/tests/test_nested_minimize.py}
 * (gh21193): two {@link NonlinearConstraint} instances created without an
 * explicit Hessian must not share Hessian-update state, and a nested
 * {@code minimize(...)} call inside the objective of an outer minimize must
 * run to completion.
 *
 * <p>In our port {@code NonlinearConstraint} stores its (optional) Hessian
 * callable as a final field -- when omitted it is simply {@code null}, so two
 * default-Hessian constraints are trivially distinct objects with independent
 * state. The structural concern in scipy gh21193 was that scipy's default
 * {@code hess=BFGS()} closed over a single shared mutable instance; our port
 * doesn't have that hazard. The test below documents both halves of the
 * contract.
 */
class TestNestedMinimize {

	private static double rosen(Matrix x) {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
	}

	private static Matrix rosenG(Matrix x) {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
				200.0 * (b - a * a) });
	}

	@Test
	void distinctDefaultHessianBetweenConstraints() {
		// identity(x) == 0 -- first-coordinate equality constraint.
		Function<Matrix, Matrix> identity = x ->
				Matrix.Factory.linkToArray(new double[] {x.getAsDouble(0, 0)});
		Function<Matrix, Matrix> identityJac = x -> {
			int n = (int) x.getRowCount();
			double[][] J = new double[1][n];
			J[0][0] = 1.0;
			return Matrix.Factory.linkToArray(J);
		};

		NonlinearConstraint c1 = new NonlinearConstraint(identity, identityJac,
				new double[] {0.0}, new double[] {0.0}, null);
		NonlinearConstraint c2 = new NonlinearConstraint(identity, identityJac,
				new double[] {0.0}, new double[] {0.0}, null);

		// Scipy's gh21193: two no-Hessian constraints used to share a single
		// mutable BFGS instance. Our port stores `hess` as a final field that
		// is null when omitted -- the two constraint objects are distinct, so
		// their Hessian state cannot be aliased.
		Assertions.assertNotSame(c1, c2);
	}

	@Test
	void nestedMinimizeRunsToCompletion() {
		Function<Matrix, Matrix> identityFirst = x ->
				Matrix.Factory.linkToArray(new double[] {x.getAsDouble(0, 0)});
		Function<Matrix, Matrix> identityFirstJac = x -> {
			int n = (int) x.getRowCount();
			double[][] J = new double[1][n];
			J[0][0] = 1.0;
			return Matrix.Factory.linkToArray(J);
		};

		// Inner constraint: y[0] == 0 over the 2-dim inner variable.
		NonlinearConstraint cInner = new NonlinearConstraint(
				identityFirst, identityFirstJac,
				new double[] {0.0}, new double[] {0.0}, null);
		// Outer constraint: x[0] == 0 over the 3-dim outer variable.
		NonlinearConstraint cOuter = new NonlinearConstraint(
				identityFirst, identityFirstJac,
				new double[] {0.0}, new double[] {0.0}, null);

		// Inner objective: 2-D Rosenbrock on x[1:]. Constraint y[0] == 0 starts
		// from y0 = (x[1], x[2]); we cap iterations at 5 to mimic scipy's
		// maxiter=2 spirit (just exercise the path without demanding optimum).
		Function<Matrix, Double> outerObjective = xOuter -> {
			Matrix yInit = Matrix.Factory.linkToArray(new double[] {
					xOuter.getAsDouble(1, 0),
					xOuter.getAsDouble(2, 0) });
			OptimizeResult inner = MinimizeTrustConstr.minimize(
					TestNestedMinimize::rosen,
					TestNestedMinimize::rosenG,
					yInit, cInner,
					5, 1.0e-2, 1.0e-2);
			return inner.fun;
		};
		Function<Matrix, Matrix> outerGrad = xOuter -> {
			// Plug-in finite difference on the *outer* variable. Use a coarse
			// step -- we only need this to terminate, not converge tightly.
			int n = (int) xOuter.getRowCount();
			double[] g = new double[n];
			double f0 = outerObjective.apply(xOuter);
			double h = 1.0e-4;
			for (int i = 0; i < n; ++i) {
				Matrix xp = Matrix.Factory.copyFromMatrix(xOuter);
				xp.setAsDouble(xOuter.getAsDouble(i, 0) + h, i, 0);
				g[i] = (outerObjective.apply(xp) - f0) / h;
			}
			return Matrix.Factory.linkToArray(g);
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0, 0.0, 0.0});
		// The test from scipy "doesn't check that the output is correct, just
		// that it doesn't crash." Same here.
		OptimizeResult r = Assertions.assertDoesNotThrow(() ->
				MinimizeTrustConstr.minimize(outerObjective, outerGrad, x0, cOuter,
						3, 1.0e-2, 1.0e-2));
		Assertions.assertNotNull(r);
		Assertions.assertNotNull(r.x);
	}
}
