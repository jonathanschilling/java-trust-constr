package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinAlg;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

class TestNonlinearConstraint {

	private static final double TOL = 1.0e-12;
	private static final double NEG_INF = Double.NEGATIVE_INFINITY;
	private static final double POS_INF = Double.POSITIVE_INFINITY;

	@Test
	void testEqualityScalar() {
		// fun(x) = x[0]^2 + x[1]^2; constraint: fun(x) == 1 (unit circle)
		java.util.function.Function<Matrix, Matrix> fun = x -> {
			double x0 = x.getAsDouble(0, 0);
			double x1 = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] { x0 * x0 + x1 * x1 });
		};
		java.util.function.Function<Matrix, Matrix> jac = x -> {
			double x0 = x.getAsDouble(0, 0);
			double x1 = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] { { 2 * x0, 2 * x1 } });
		};
		NonlinearConstraint c = new NonlinearConstraint(fun, jac, new double[] {1.0}, new double[] {1.0});
		Assertions.assertEquals(1, c.nEq());
		Assertions.assertEquals(0, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {0.6, 0.8});
		// f(x) = 0.36 + 0.64 = 1; constrEq = 1 - 1 = 0
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.0}, LinAlg.col(c.constrEq(x)), TOL);
		// jacEq = [2*0.6, 2*0.8] = [1.2, 1.6]
		MinervaAssertions.assertArrayRelAbsEquals(new double[][] { {1.2, 1.6} }, c.jacEq(x).toDoubleArray(), TOL);
	}

	@Test
	void testInequalityUpper() {
		// fun(x) = x[0]; constraint: fun(x) <= 5
		java.util.function.Function<Matrix, Matrix> fun = x ->
				Matrix.Factory.linkToArray(new double[] { x.getAsDouble(0, 0) });
		java.util.function.Function<Matrix, Matrix> jac = x ->
				Matrix.Factory.linkToArray(new double[][] { { 1.0 } });
		NonlinearConstraint c = new NonlinearConstraint(fun, jac, new double[] {NEG_INF}, new double[] {5.0});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(1, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {7.0});
		// constrIneq = +1*(7 - 5) = 2  (positive => violated)
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {2.0}, LinAlg.col(c.constrIneq(x)), TOL);
	}

	@Test
	void testTwoSidedRejected() {
		java.util.function.Function<Matrix, Matrix> fun = x ->
				Matrix.Factory.linkToArray(new double[] { x.getAsDouble(0, 0) });
		Assertions.assertThrows(UnsupportedOperationException.class,
				() -> new NonlinearConstraint(fun, fun, new double[] {0.0}, new double[] {1.0}));
	}
}
