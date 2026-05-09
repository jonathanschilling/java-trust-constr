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
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Rosenbrock with two-sided bounds (matlab {@code fmincon} docs / scipy
 * {@code BoundedRosenbrock}):
 *
 * <pre>
 *     minimize  100*(y - x^2)^2 + (1 - x)^2
 *     s.t.  -2 &lt;= x &lt;= 0
 *           0  &lt;= y &lt;= 2
 * </pre>
 *
 * Optimum (scipy reference): {@code x* ~ (0, 0)}, {@code f* ~ 1}. The
 * unconstrained Rosenbrock minimum (1, 1) is outside the box; the constrained
 * optimum is the corner of the feasible box closest to (1, 1) along the
 * Rosenbrock valley.
 */
class TestBoundedRosenbrock {

	@Test
	void rosenbrockUnderBoxBounds() {
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a) });
		};
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ 2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a },
					{ -400.0 * a, 200.0 } });
		};

		Bounds box = new Bounds(
				Matrix.Factory.linkToArray(new double[] {-2.0, 0.0}),
				Matrix.Factory.linkToArray(new double[] {0.0, 2.0}),
				false);
		LinearConstraint asConstraint = LinearConstraint.fromBounds(box);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-0.2, 0.2});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0,
				asConstraint, 2000, 1.0e-8, 1.0e-8);

		// scipy converges to x ~ (0, 0), f ~ 1.
		Assertions.assertEquals(0.0, r.x.getAsDouble(0, 0), 5.0e-3);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 5.0e-3);
		Assertions.assertEquals(1.0, r.fun, 1.0e-2);
	}
}
