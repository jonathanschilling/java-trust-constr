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
package de.labathome.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Function;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.LagrangianHessian;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Unit tests for {@link LagrangianHessian} -- the Java port's counterpart to
 * scipy's {@code _trustregion_constr/minimize_trustregion_constr.py:LagrangianHessian}.
 */
class TestLagrangianHessian {

	private static final double TOL = 1e-12;

	@Test
	void appliesObjectiveOnlyWhenNoConstraintHessian() {
		// H_obj = diag(2, 3). Constraint hess returns null.
		Function<Matrix, Matrix> objHess = x -> DenseMatrix.fromRows(
				new double[][] {{2, 0}, {0, 3}});
		LagrangianHessian.ConstraintHessian noHess = (x, vEq, vIneq) -> null;
		LagrangianHessian lh = new LagrangianHessian(2, objHess, noHess);

		Matrix x = DenseMatrix.column(1.0, 1.0);
		LinearOperator op = lh.apply(x, new double[0], new double[0]);
		Matrix p = DenseMatrix.column(1.0, 0.0);
		// H @ p = (2, 0)
		Matrix Hp = op.apply(p);
		RelAbsAssertions.assertRelAbsEquals(2.0, Hp.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(0.0, Hp.getAsDouble(1, 0), TOL);
	}

	@Test
	void sumsObjectiveAndConstraintHessians() {
		// H_obj = I, H_constr(x, vEq) = vEq[0] * diag(4, 4).
		Function<Matrix, Matrix> objHess = x -> DenseMatrix.fromRows(
				new double[][] {{1, 0}, {0, 1}});
		LagrangianHessian.ConstraintHessian cHess = (x, vEq, vIneq) -> {
			double v = vEq[0];
			return DenseMatrix.fromRows(new double[][] {{4 * v, 0}, {0, 4 * v}});
		};
		LagrangianHessian lh = new LagrangianHessian(2, objHess, cHess);

		Matrix x = DenseMatrix.column(0.0, 0.0);
		double[] vEq = {0.5};
		LinearOperator op = lh.apply(x, vEq, new double[0]);

		// H = I + 0.5*4*I = I + 2*I = 3*I
		Matrix p = DenseMatrix.column(1.0, 1.0);
		Matrix Hp = op.apply(p);
		RelAbsAssertions.assertRelAbsEquals(3.0, Hp.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(3.0, Hp.getAsDouble(1, 0), TOL);

		assertEquals(2, lh.n());
	}

	@Test
	void worksForLargerProblem() {
		final int n = 4;
		Function<Matrix, Matrix> objHess = x -> Matrix.Factory.eye(n, n);
		LagrangianHessian.ConstraintHessian noHess = (xx, vEq, vIneq) -> null;
		LagrangianHessian lh = new LagrangianHessian(n, objHess, noHess);

		Matrix x = DenseMatrix.column(0.0, 0.0, 0.0, 0.0);
		LinearOperator op = lh.apply(x, new double[0], new double[0]);
		Matrix p = DenseMatrix.column(1.0, 2.0, 3.0, 4.0);
		Matrix Hp = op.apply(p);
		// H = I -> Hp = p
		for (int i = 0; i < n; ++i) {
			RelAbsAssertions.assertRelAbsEquals(p.getAsDouble(i, 0),
					Hp.getAsDouble(i, 0), TOL, "row " + i);
		}
	}
}
