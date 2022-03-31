package de.labathome.optimization;

import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;

class TestApproxDerivativesDense {

	/** scalar function of a scalar argument */
	ToDoubleBiFunction<Matrix, Object> funScalarScalar = (Matrix x, Object args) -> {
		return Math.sinh(x.doubleValue());
	};

	/** Jacobian of the scalar function with scalar argument */
	BiFunction<Matrix, Object, Matrix> jacScalarScalar = (Matrix x, Object args) -> {
		return Matrix.Factory.linkToArray(new double[] {Math.cosh(x.doubleValue())});
	};

	/** vector-valued function of a scalar argument */
	BiFunction<Matrix, Object, Matrix> funScalarVector = (Matrix x, Object args) -> {
		double x0 = x.doubleValue();
		return Matrix.Factory.linkToArray(new double[] {
				x0 * x0,
				Math.tan(x0),
				Math.exp(x0)
		});
	};

	/** Jacobian of the vector-valued function of a scalar argument */
	BiFunction<Matrix, Object, Matrix> jacScalarVector = (Matrix x, Object args) -> {
		double x0 = x.doubleValue();
		double cosX0 = Math.cos(x0);
		return Matrix.Factory.linkToArray(new double[] {
				2.0 * x0,
				1.0/(cosX0 * cosX0),
				Math.exp(x0)
		});
	};

	/** scalar function of a vector argument */
	BiFunction<Matrix, Object, Matrix> funVectorScalar = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				Math.sin(x0 * x1) * Math.log(x0)
		});
	};

	/** Jacobian of the scalar function of a vector argument */
	BiFunction<Matrix, Object, Matrix> jacVectorScalar = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x1 * Math.cos(x0 * x1) * Math.log(x0) + Math.sin(x0 * x1) / x0,
				x0 * Math.cos(x0 * x1) * Math.log(x0)
		});
	};

	/** vector-valued function of a vector argument */
	BiFunction<Matrix, Object, Matrix> funVectorVector = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x0 * Math.sin(x1),
				x1 * Math.cos(x0),
				x0*x0*x0 / Math.sqrt(x1)
		});
	};

	/** Jacobian of the vector-valued function of a vector argument */
	BiFunction<Matrix, Object, Matrix> jacVectorVector = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
			{Math.sin(x1), x0 * Math.cos(x1)},
			{-x1 * Math.sin(x0), Math.cos(x0)},
			{3.0*x0*x0 / Math.sqrt(x1), -0.5 * x0*x0*x0 * Math.pow(x1, -1.5)}
		});
	};

	/** parameterized vector-valued function of a vector argument */
	BiFunction<Matrix, Object, Matrix> funParameterized = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		double c0 = Double.NaN;
		double c1 = 1.0;
		if (args != null && args instanceof double[]) {
			double[] c = (double[]) args;
			c0 = c[0];
			if (c.length > 1) {
				c1 = c[1];
			}
		}

		return Matrix.Factory.linkToArray(new double[] {
				Math.exp(c0 * x0),
				Math.exp(c1 * x1)
		});
	};

	/** Jacobian of the parameterized vector-valued function of a vector argument */
	BiFunction<Matrix, Object, Matrix> jacParameterized = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		double c0 = Double.NaN;
		double c1 = 0.1; // dfferent default value to test if all arguments are actually passed
		if (args != null && args instanceof double[]) {
			double[] c = (double[]) args;
			c0 = c[0];
			if (c.length > 1) {
				c1 = c[1];
			}
		}

		return Matrix.Factory.linkToArray(new double[][] {
			{c0 * Math.exp(c0 * x0), 0.0},
			{0.0, c1 * Math.exp(c1 * x1)},
		});
	};

	/** scalar function of a scalar argument that is only non-NaN in a small region around 0 */
	ToDoubleBiFunction<Matrix, Object> funWithNaN = (Matrix x, Object args) -> {
		if (Math.abs(x.doubleValue()) <= 1.0e-8) {
			return x.doubleValue();
		} else {
			return Double.NaN;
		}
	};

	/** Jacobian of the scalar function of a scalar argument that is only non-NaN in a small region around 0 */
	BiFunction<Matrix, Object, Matrix> jacWithNaN = (Matrix x, Object args) -> {
		if (Math.abs(x.doubleValue()) <= 1.0e-8) {
			return Matrix.Factory.ones(1, 1);
		} else {
			return Matrix.Factory.ones(1, 1).times(Double.NaN);
		}
	};

	/** vector-valued function of a vector argument where the Jacobian can become zero */
	BiFunction<Matrix, Object, Matrix> funZeroJacobian = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				x0 * x1,
				Math.cos(x0 * x1)
		});
	};

	/** Jacobian of the vector-valued function of a vector argument where the Jacobian can become zero */
	BiFunction<Matrix, Object, Matrix> jacZeroJacobian = (Matrix x, Object args) -> {
		double x0 = x.getAsDouble(0, 0);
		double x1 = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
				{x1, x0},
				{-x1 * Math.sin(x0 * x1), -x0 * Math.sin(x0 *x1)}
		});
	};

	@Test
	void testScalarScalar() {




	}



}
