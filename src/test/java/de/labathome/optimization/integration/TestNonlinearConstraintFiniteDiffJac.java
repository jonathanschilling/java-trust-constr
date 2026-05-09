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
 * Tests the {@link NonlinearConstraint} constructor that omits the analytic
 * Jacobian and computes one via 2-point finite differences on each call --
 * mirroring scipy's behaviour when {@code jac} is omitted.
 */
class TestNonlinearConstraintFiniteDiffJac {

	@Test
	void unitHyperbolaConvergesWithFdJacobian() {
		// Same problem as TestUnitHyperbola but with the constraint Jacobian
		// computed by finite differences instead of supplied analytically.
		// Optimum on the unit hyperbola: |x|=1, y=0, f=1.
		Function<Matrix, Double> fun = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> grad = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> hess = x ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				- x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});

		// No-Jacobian constructor: FD is computed internally per evaluation.
		NonlinearConstraint c = new NonlinearConstraint(cFun,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5, 0.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, hess, x0, c,
				1000, 1.0e-7, 1.0e-7);

		// FD is noisier than analytic; loosen tolerance vs the analytic test.
		Assertions.assertEquals(1.0, Math.abs(r.x.getAsDouble(0, 0)), 1.0e-3);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.fun, 1.0e-3);
	}

	@Test
	void fdJacobianCallEvaluatesFunNplusOneTimes() {
		// One-shot sanity check: the FD Jacobian path calls fun(x) once for the
		// baseline plus n more times for the perturbed columns (n = x.size()).
		final int[] callCount = {0};
		Function<Matrix, Matrix> cFun = x -> {
			callCount[0]++;
			return Matrix.Factory.linkToArray(new double[] {
					x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
					+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});
		};
		NonlinearConstraint c = new NonlinearConstraint(cFun,
				new double[] {1.0}, new double[] {1.0});

		Matrix x = Matrix.Factory.linkToArray(new double[] {1.0, 1.0});
		// jacEq triggers FD: 1 baseline + 2 perturbations = 3 calls.
		Matrix Jeq = c.jacEq(x);
		Assertions.assertEquals(1, Jeq.getRowCount());
		Assertions.assertEquals(2, Jeq.getColumnCount());
		// Analytic gradient at (1,1) for fun = x^2 + y^2 is (2, 2).
		Assertions.assertEquals(2.0, Jeq.getAsDouble(0, 0), 1.0e-5);
		Assertions.assertEquals(2.0, Jeq.getAsDouble(0, 1), 1.0e-5);
		Assertions.assertEquals(3, callCount[0],
				"FD Jacobian on n=2 should call fun 1+n=3 times; got " + callCount[0]);
	}
}
