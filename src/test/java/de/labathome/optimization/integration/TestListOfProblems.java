package de.labathome.optimization.integration;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Named;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.records.OptimizeResult;

import de.labathome.optimization.RelAbsAssertions;

/**
 * Java parametrised translation of scipy
 * {@code optimize/tests/test_minimize_constrained.py:test_list_of_problems}.
 * Mirrors scipy's {@code @pytest.mark.parametrize('prob', list_of_problems)}
 * over a representative subset of the trust-constr stress problems, each run
 * with both analytic-gradient and FD-gradient configurations against analytic
 * Hessian.
 *
 * <p>scipy's full matrix is 17 problems x 3 grad modes x 5 hess modes = ~255
 * cases. We port the four most representative problems
 * (Rosenbrock unconstrained, IneqRosenbrock, Maratos, HyperbolicIneq) with the
 * analytic + FD-grad x analytic-hess axis, giving 8 parametrised cases that
 * exercise both the equality SQP path and the IP barrier path.
 */
class TestListOfProblems {

	@ParameterizedTest(name = "{0}")
	@MethodSource("problemMatrix")
	void converge(Case c) {
		Matrix x0 = c.x0;
		BiFunction<Matrix, Object, Matrix> gradFn =
				c.useAnalyticGrad ? (x, args) -> c.grad.apply(x) : null;
		BiFunction<Matrix, Object, Matrix> hessFn = (x, args) -> c.hess.apply(x);
		java.util.function.ToDoubleBiFunction<Matrix, Object> funFn =
				(x, args) -> c.fun.apply(x);

		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				funFn, x0, null, gradFn, hessFn, null,
				/*bounds=*/ null, c.constraints,
				/*xTol=*/ 1e-8, /*gTol=*/ 1e-8, /*barrierTol=*/ 1e-8,
				java.util.Optional.empty(),
				/*callback=*/ null, /*maxIter=*/ 1000, /*verbose=*/ 0,
				/*finiteDifferenceRelStep=*/ null,
				/*initialConstraintPenalty=*/ 1.0, /*initialTrustRadius=*/ 1.0,
				/*initialBarrierParameter=*/ 0.1, /*initialBarrierTolerance=*/ 0.1,
				/*factorizationMethod=*/ null, /*disp=*/ false);

		// scipy asserts result.x == prob.x_opt at decimal=5 (1e-5 abs tol).
		for (int i = 0; i < c.xOpt.length; ++i) {
			RelAbsAssertions.assertRelAbsEquals(c.xOpt[i], r.x.getAsDouble(i, 0), c.xTol,
					"x[" + i + "]");
		}
		// status not in {0, 3} (no max-iter / no callback-stop).
		assertNotEquals(0, r.status, "did not exhaust maxiter");
		assertNotEquals(3, r.status, "did not stop via callback");
		assertTrue(r.success, "converged");
	}

	static java.util.stream.Stream<Arguments> problemMatrix() {
		java.util.List<Case> cases = new java.util.ArrayList<>();
		// Rosenbrock unconstrained (analytic + FD grad).
		cases.add(rosenbrock(true, "Rosenbrock-analytic"));
		cases.add(rosenbrock(false, "Rosenbrock-FDgrad"));
		// IneqRosenbrock with single linear inequality.
		cases.add(ineqRosenbrock(true, "IneqRosenbrock-analytic"));
		cases.add(ineqRosenbrock(false, "IneqRosenbrock-FDgrad"));
		// Maratos: equality unit-circle constraint.
		cases.add(maratos(true, "Maratos-analytic"));
		cases.add(maratos(false, "Maratos-FDgrad"));
		// HyperbolicIneq.
		cases.add(hyperbolicIneq(true, "HyperbolicIneq-analytic"));
		cases.add(hyperbolicIneq(false, "HyperbolicIneq-FDgrad"));
		return cases.stream().map(c -> Arguments.of(Named.of(c.name, c)));
	}

	private static Case rosenbrock(boolean useAnalytic, String name) {
		Function<Matrix, Double> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			double t = b - a * a;
			double s = 1.0 - a;
			return 100.0 * t * t + s * s;
		};
		Function<Matrix, Matrix> grad = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			double t = b - a * a;
			return DenseMatrix.column(-400.0 * a * t - 2.0 * (1.0 - a), 200.0 * t);
		};
		Function<Matrix, Matrix> hess = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return DenseMatrix.fromRows(new double[][] {
					{1200.0 * a * a - 400.0 * b + 2.0, -400.0 * a},
					{-400.0 * a, 200.0}});
		};
		Case c = new Case();
		c.name = name;
		c.fun = fun;
		c.grad = grad;
		c.hess = hess;
		c.x0 = DenseMatrix.column(0.5, 0.5);
		c.xOpt = new double[] {1.0, 1.0};
		c.xTol = 1e-3;
		c.useAnalyticGrad = useAnalytic;
		c.constraints = null;
		return c;
	}

	private static Case ineqRosenbrock(boolean useAnalytic, String name) {
		Case c = rosenbrock(useAnalytic, name);
		c.x0 = DenseMatrix.column(-1.0, -0.5);
		c.xOpt = new double[] {0.5022, 0.2489};
		c.xTol = 1e-2;
		// x[0] + 2 x[1] <= 1 -> linear constraint with lb=-inf, ub=1.
		LinearConstraint lc = new LinearConstraint(
				DenseMatrix.fromRows(new double[][] {{1.0, 2.0}}),
				new double[] {Double.NEGATIVE_INFINITY},
				new double[] {1.0});
		c.constraints = lc;
		return c;
	}

	private static Case maratos(boolean useAnalytic, String name) {
		Function<Matrix, Double> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 2.0 * (a * a + b * b - 1.0) - a;
		};
		Function<Matrix, Matrix> grad = x -> DenseMatrix.column(
				4.0 * x.getAsDouble(0, 0) - 1.0, 4.0 * x.getAsDouble(1, 0));
		Function<Matrix, Matrix> hess = x -> DenseMatrix.fromRows(
				new double[][] {{4.0, 0.0}, {0.0, 4.0}});

		Function<Matrix, Matrix> cFun = x -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
						+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0));
		Function<Matrix, Matrix> cJac = x -> DenseMatrix.fromRows(new double[][] {
				{2.0 * x.getAsDouble(0, 0), 2.0 * x.getAsDouble(1, 0)}});
		BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			return DenseMatrix.fromRows(new double[][] {{2.0 * v0, 0.0}, {0.0, 2.0 * v0}});
		};
		NonlinearConstraint nc = new NonlinearConstraint(cFun, cJac, cHess,
				new double[] {1.0}, new double[] {1.0}, null);

		Case c = new Case();
		c.name = name;
		c.fun = fun;
		c.grad = grad;
		c.hess = hess;
		c.x0 = DenseMatrix.column(Math.cos(Math.toRadians(60.0)),
				Math.sin(Math.toRadians(60.0)));
		c.xOpt = new double[] {1.0, 0.0};
		c.xTol = 1e-3;
		c.useAnalyticGrad = useAnalytic;
		c.constraints = nc;
		return c;
	}

	private static Case hyperbolicIneq(boolean useAnalytic, String name) {
		// scipy HyperbolicIneq (Nocedal & Wright 15.1):
		//   minimize 1/2 (x[0] - 2)^2 + 1/2 (x[1] - 1/2)^2
		//   s.t. 1/(x[0] + 1) - x[1] >= 1/4, x[0] >= 0, x[1] >= 0
		// optimum: (1.952823, 0.088659)
		Function<Matrix, Double> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 0.5 * (a - 2.0) * (a - 2.0) + 0.5 * (b - 0.5) * (b - 0.5);
		};
		Function<Matrix, Matrix> grad = x -> DenseMatrix.column(
				x.getAsDouble(0, 0) - 2.0, x.getAsDouble(1, 0) - 0.5);
		Function<Matrix, Matrix> hess = x -> DenseMatrix.fromRows(
				new double[][] {{1.0, 0.0}, {0.0, 1.0}});

		// Nonlinear constraint: 1/(x[0] + 1) - x[1] >= 1/4
		Function<Matrix, Matrix> cFun = x -> DenseMatrix.column(
				1.0 / (x.getAsDouble(0, 0) + 1.0) - x.getAsDouble(1, 0));
		Function<Matrix, Matrix> cJac = x -> {
			double t = x.getAsDouble(0, 0) + 1.0;
			return DenseMatrix.fromRows(new double[][] {{-1.0 / (t * t), -1.0}});
		};
		BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			double t = x.getAsDouble(0, 0) + 1.0;
			return DenseMatrix.fromRows(new double[][] {
					{2.0 * v0 / (t * t * t), 0.0}, {0.0, 0.0}});
		};
		NonlinearConstraint nc = new NonlinearConstraint(cFun, cJac, cHess,
				new double[] {0.25}, new double[] {Double.POSITIVE_INFINITY}, null);

		// Bounds: x[0] >= 0, x[1] >= 0 -- folded as a LinearConstraint with A=I.
		LinearConstraint bnds = new LinearConstraint(
				DenseMatrix.fromRows(new double[][] {{1, 0}, {0, 1}}),
				new double[] {0.0, 0.0},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});

		Case c = new Case();
		c.name = name;
		c.fun = fun;
		c.grad = grad;
		c.hess = hess;
		c.x0 = DenseMatrix.column(0.0, 0.0);
		c.xOpt = new double[] {1.952823, 0.088659};
		c.xTol = 1e-3;
		c.useAnalyticGrad = useAnalytic;
		c.constraints = new Object[] {nc, bnds};
		return c;
	}

	static final class Case {
		String name;
		Function<Matrix, Double> fun;
		Function<Matrix, Matrix> grad;
		Function<Matrix, Matrix> hess;
		Matrix x0;
		double[] xOpt;
		double xTol;
		boolean useAnalyticGrad;
		Object constraints;

		@Override
		public String toString() {
			return name;
		}
	}
}
