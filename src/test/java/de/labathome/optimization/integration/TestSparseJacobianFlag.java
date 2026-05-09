package de.labathome.optimization.integration;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.SparsityForcedConstraint;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

/**
 * Tests for the explicit {@code Optional<Boolean> sparseJacobian} parameter
 * to {@link MinimizeTrustConstr#minimizeTrustConstr}. When set, every
 * constraint Jacobian is forced to the requested representation via
 * {@link SparsityForcedConstraint} — overriding the auto-detect logic.
 */
class TestSparseJacobianFlag {

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
	void sparsityForcedConstraintWrapperUnitChecks() {
		// Direct unit check on SparsityForcedConstraint: dense source forced
		// to sparse, and sparse source forced to dense.
		Matrix Adense = Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}});
		LinearConstraint dense = new LinearConstraint(Adense,
				new double[] {2.0}, new double[] {2.0});
		SparsityForcedConstraint forced = new SparsityForcedConstraint(dense, /*wantSparse=*/ true);

		Matrix x = Matrix.Factory.linkToArray(new double[] {1.0, 1.0});
		Matrix Jeq = forced.jacEq(x);
		Assertions.assertTrue(Jeq.isSparse(),
				"Forced-sparse wrapper should produce sparse jacEq even from dense A");
		Assertions.assertEquals(1.0, Jeq.getAsDouble(0, 0), 1.0e-12);
		Assertions.assertEquals(1.0, Jeq.getAsDouble(0, 1), 1.0e-12);

		// Now sparse source forced to dense.
		SparseMatrix Asp = SparseMatrix.Factory.zeros(1, 2);
		Asp.setAsDouble(2.0, 0, 0);
		Asp.setAsDouble(3.0, 0, 1);
		LinearConstraint sparse = new LinearConstraint(Asp,
				new double[] {0.0}, new double[] {0.0});
		SparsityForcedConstraint forcedDense = new SparsityForcedConstraint(sparse, /*wantSparse=*/ false);
		Matrix Jeq2 = forcedDense.jacEq(x);
		Assertions.assertFalse(Jeq2.isSparse(),
				"Forced-dense wrapper should produce dense jacEq even from sparse A");
		Assertions.assertEquals(2.0, Jeq2.getAsDouble(0, 0), 1.0e-12);
		Assertions.assertEquals(3.0, Jeq2.getAsDouble(0, 1), 1.0e-12);
	}

	@Test
	void sparseJacobianTrueForcesSparseEndToEnd() {
		// Hyperplane Rosenbrock with a *dense* A, but the user asks for
		// sparseJacobian=True — the orchestrator wraps the constraint so
		// jacEq is sparse. Algorithm should still converge.
		Matrix Adense = Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}});
		LinearConstraint eq = new LinearConstraint(Adense,
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.of(true),                  // sparseJacobian = True
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		Assertions.assertTrue(r.success);
	}

	@Test
	void sparseJacobianFalseForcesDenseEndToEnd() {
		// Hyperplane Rosenbrock with a *sparse* A but the user asks for
		// sparseJacobian=False — the orchestrator wraps the constraint so
		// jacEq is dense. Algorithm should still converge.
		SparseMatrix Asp = SparseMatrix.Factory.zeros(1, 2);
		Asp.setAsDouble(1.0, 0, 0);
		Asp.setAsDouble(1.0, 0, 1);
		LinearConstraint eq = new LinearConstraint(Asp,
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.of(false),                 // sparseJacobian = False
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		Assertions.assertTrue(r.success);
	}
}
