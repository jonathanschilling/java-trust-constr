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

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * End-to-end tests for {@link MinimizeTrustConstr#minimizeTrustConstr}, the
 * scipy-shape entry point. The body adapts the broad signature
 * ({@code BiFunction<Matrix, Object, ...>}, {@link Bounds}, {@code hessp},
 * etc.) to the convenience overloads that carry the actual SQP / IP machinery.
 *
 * <p>Each test exercises one of the three derivative-supplied dispatch
 * branches (analytic-everything, BFGS-fallback, FD+BFGS-fallback) on a
 * representative constraint shape. {@code bounds} is also exercised since the
 * adapter folds it into the constraint set as a {@link LinearConstraint}.
 */
class TestMinimizeTrustConstrFullShape {

	private static final ToDoubleBiFunction<Matrix, Object> rosenFun = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenGrad = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
				200.0 * (b - a * a)});
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenHess = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
				{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
				{-400.0 * a, 200.0}});
	};

	@Test
	void analyticEverythingUnconstrained() {
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null,                       // hessp
				null,                       // bounds
				null,                       // constraints
				1.0e-10, 1.0e-10, 1.0e-10,  // xtol, gtol, barrierTol
				Optional.empty(),           // sparseJacobian
				null,                       // callback
				1000,                       // maxIter
				0,                          // verbose
				null,                       // finiteDifferenceRelStep
				1.0, 1.0, 0.1, 0.1,         // initialPenalty / trustRadius / barrierParameter / barrierTolerance
				null,                       // factorizationMethod
				false);                     // disp

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-5);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-5);
		Assertions.assertEquals(0.0, r.fun, 1.0e-10);
	}

	@Test
	void analyticEverythingWithLinearConstraint() {
		// Rosenbrock under x[0] + x[1] = 2; optimum (1, 1), f = 0.
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null,
				eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
	}

	@Test
	void bfgsFallbackWithBoundsOnly() {
		// Adapter folds Bounds into the constraint set as a LinearConstraint
		// over the identity Jacobian. Test: bounds-only Rosenbrock with
		// BFGS-Hessian, optimum at (1, 1) since interior.
		Bounds b = new Bounds(
				Matrix.Factory.linkToArray(new double[] {-2.0, -2.0}),
				Matrix.Factory.linkToArray(new double[] {2.0, 2.0}),
				false);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad,
				null,                       // hess: triggers BFGS fallback
				null, b,
				null,
				1.0e-8, 1.0e-8, 1.0e-8,
				Optional.empty(),
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-3);
	}

	@Test
	void fdGradPlusBfgsHessOnNonlinearConstraint() {
		// Most hands-off entry: only the objective. Adapter strips args and
		// dispatches to the (fun, x0, constraint) overload -- which builds
		// the FD gradient + BFGS Hessian internally.
		ToDoubleBiFunction<Matrix, Object> q = (x, args) ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);

		// Constraint x^2 - y^2 >= 1 (unit hyperbola).
		Function<Matrix, Matrix> cFun = x ->
				Matrix.Factory.linkToArray(new double[] {
						x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
						- x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> cJac = x ->
				Matrix.Factory.linkToArray(new double[][] {
						{2 * x.getAsDouble(0, 0), -2 * x.getAsDouble(1, 0)}});
		NonlinearConstraint c = new NonlinearConstraint(cFun, cJac,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5, 0.0});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				q, x0, null,
				null,                       // grad: triggers FD fallback
				null,                       // hess: triggers BFGS fallback
				null, null,
				c,
				1.0e-7, 1.0e-7, 1.0e-7,
				Optional.empty(),
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, Math.abs(r.x.getAsDouble(0, 0)), 5.0e-2);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 5.0e-2);
		Assertions.assertEquals(1.0, r.fun, 1.0e-2);
	}

	@Test
	void argsParameterReachesFunGradHess() {
		// Verify that the Object `args` parameter flows through to fun/grad/hess.
		// Use args to carry a coefficient k, with f(x; k) = k * (x[0]^2 + x[1]^2).
		// Constraint x[0] + x[1] = 2 -> optimum (1, 1), f = 2 * k.
		// Picking k = 5 makes the test depend on args being non-null.
		final double k = 5.0;
		ToDoubleBiFunction<Matrix, Object> fun = (x, a) -> {
			double coef = (Double) a;
			return coef * (x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
					+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0));
		};
		BiFunction<Matrix, Object, Matrix> grad = (x, a) -> {
			double coef = (Double) a;
			return Matrix.Factory.linkToArray(new double[] {
					2 * coef * x.getAsDouble(0, 0),
					2 * coef * x.getAsDouble(1, 0)});
		};
		BiFunction<Matrix, Object, Matrix> hess = (x, a) -> {
			double coef = (Double) a;
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * coef, 0}, {0, 2 * coef}});
		};

		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				fun, x0, k,            // args = k
				grad, hess,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				null,
				200, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		// f = k * (1+1) = 2k = 10. If args weren't propagated, args would be
		// null and the cast would NPE -- or k=0 default would converge to f=0.
		Assertions.assertEquals(2.0 * k, r.fun, 1.0e-8,
				"args should reach fun: f should be 2k=" + (2.0 * k) + " but was " + r.fun);
	}

	@Test
	void analyticEverythingWithMultiConstraint() {
		// Rosenbrock with mixed equality and inequality from EqIneqRosenbrock.
		// 2x + y = 1, x + 2y <= 1. Adapter passes Object[] through to the
		// multi-constraint overload.
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{2.0, 1.0}}),
				new double[] {1.0}, new double[] {1.0});
		LinearConstraint ineq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 2.0}}),
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.4, 0.2});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null,
				new Object[] {eq, ineq},
				1.0e-8, 1.0e-8, 1.0e-8,
				Optional.empty(),
				null,
				2000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(0.41494, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(0.17011, r.x.getAsDouble(1, 0), 1.0e-2);
	}
}
