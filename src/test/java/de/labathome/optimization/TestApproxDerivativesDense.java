package de.labathome.optimization;

import java.util.function.BiFunction;
import java.util.function.DoubleFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.NumDiff;
import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

class TestApproxDerivativesDense {

	/** scalar function of a scalar argument */
	DoubleUnaryOperator funScalarScalar = (double x) -> {
		return Math.sinh(x);
	};

	/** Jacobian of the scalar function with scalar argument */
	DoubleUnaryOperator jacScalarScalar = (double x) -> {
		return Math.cosh(x);
	};

	/** vector-valued function of a scalar argument */
	Function<Matrix, Matrix> funScalarVector = (Matrix x) -> {
		double x0 = x.doubleValue();
		return Matrix.Factory.linkToArray(new double[] {
				x0 * x0,
				Math.tan(x0),
				Math.exp(x0)
		});
	};

	/** Jacobian of the vector-valued function of a scalar argument */
	Function<Matrix, Matrix> jacScalarVector = (Matrix x) -> {
		double x0 = x.doubleValue();
		double cosX0 = Math.cos(x0);
		return Matrix.Factory.linkToArray(new double[] {
				2.0 * x0,
				1.0/(cosX0 * cosX0),
				Math.exp(x0)
		});
	};

	/** scalar function of a vector argument */
	Function<Matrix, Matrix> funVectorScalar = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				Math.sin(x0 * x1) * Math.log(x0)
		});
	};

	/** Jacobian of the scalar function of a vector argument */
	Function<Matrix, Matrix> jacVectorScalar = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x1 * Math.cos(x0 * x1) * Math.log(x0) + Math.sin(x0 * x1) / x0,
				x0 * Math.cos(x0 * x1) * Math.log(x0)
		});
	};

	/** vector-valued function of a vector argument */
	Function<Matrix, Matrix> funVectorVector = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x0 * Math.sin(x1),
				x1 * Math.cos(x0),
				x0*x0*x0 / Math.sqrt(x1)
		});
	};

	/** Jacobian of the vector-valued function of a vector argument */
	Function<Matrix, Matrix> jacVectorVector = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
			{Math.sin(x1), x0 * Math.cos(x1)},
			{-x1 * Math.sin(x0), Math.cos(x0)},
			{3.0*x0*x0 / Math.sqrt(x1), -0.5 * x0*x0*x0 * Math.pow(x1, -1.5)}
		});
	};

	/** parameterized vector-valued function of a vector argument */
	BiFunction<Matrix, double[], Matrix> funParameterized = (Matrix x, double[] c) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		double c0 = c[0];
		double c1 = 1.0;
		if (c.length > 1) {
			c1 = c[1];
		}

		return Matrix.Factory.linkToArray(new double[] {
				Math.exp(c0 * x0),
				Math.exp(c1 * x1)
		});
	};

	/** Jacobian of the parameterized vector-valued function of a vector argument */
	BiFunction<Matrix, double[], Matrix> jacParameterized = (Matrix x, double[] c) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		double c0 = c[0];
		double c1 = 0.1; // dfferent default value to test if all arguments are actually passed
		if (c.length > 1) {
			c1 = c[1];
		}

		return Matrix.Factory.linkToArray(new double[][] {
			{c0 * Math.exp(c0 * x0), 0.0},
			{0.0, c1 * Math.exp(c1 * x1)},
		});
	};

	/** scalar function of a scalar argument that is only non-NaN in a small region around 0 */
	ToDoubleFunction<Matrix> funWithNaN = (Matrix x) -> {
		if (Math.abs(x.doubleValue()) <= 1.0e-8) {
			return x.doubleValue();
		} else {
			return Double.NaN;
		}
	};

	/** Jacobian of the scalar function of a scalar argument that is only non-NaN in a small region around 0 */
	Function<Matrix, Matrix> jacWithNaN = (Matrix x) -> {
		if (Math.abs(x.doubleValue()) <= 1.0e-8) {
			return Matrix.Factory.ones(1, 1);
		} else {
			return Matrix.Factory.ones(1, 1).times(Double.NaN);
		}
	};

	/** vector-valued function of a vector argument where the Jacobian can become zero */
	Function<Matrix, Matrix> funZeroJacobian = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x0 * x1,
				Math.cos(x0 * x1)
		});
	};

	/** Jacobian of the vector-valued function of a vector argument where the Jacobian can become zero */
	Function<Matrix, Matrix> jacZeroJacobian = (Matrix x) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
				{x1, x0},
				{-x1 * Math.sin(x0 * x1), -x0 * Math.sin(x0 *x1)}
		});
	};

	@Test
	void testScalarScalar() {
		final double x0 = 1.0;

		double jacDiff2 = NumDiff.approxDerivative(funScalarScalar, x0, FiniteDifferenceOptions.FACTORY.method(FiniteDifferenceMethod.TWO_POINT).build());
		double jacDiff3 = NumDiff.approxDerivative(funScalarScalar, x0, FiniteDifferenceOptions.FACTORY.build());

		double jacTrue = jacScalarScalar.applyAsDouble(x0);

		MinervaAssertions.assertRelAbsEquals(jacTrue, jacDiff2, 1.0e-6);
		MinervaAssertions.assertRelAbsEquals(jacTrue, jacDiff3, 1.0e-9);
	}



}
