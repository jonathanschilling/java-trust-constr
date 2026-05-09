package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

/**
 * Tests that {@link LinearConstraint} preserves sparsity end-to-end when the
 * caller provides a sparse {@link Matrix} for {@code A}.
 *
 * <p>The contract: when {@code A.isSparse() == true},
 * {@link LinearConstraint#jacEq} and {@link LinearConstraint#jacIneq} return
 * sparse {@link Matrix} instances (rather than densifying via
 * {@code Matrix.Factory.zeros}), which lets
 * {@code Projections.projections} route through the AUGMENTED_SYSTEM
 * factorization. Convergence and final values must match the dense path.
 */
class TestSparseLinearConstraint {

	@Test
	void sparseAjacEqRetainsSparsity() {
		// Sparse 2x4 matrix: pick out first and third rows for an equality.
		SparseMatrix A = SparseMatrix.Factory.zeros(2, 4);
		A.setAsDouble(1.0, 0, 0);
		A.setAsDouble(2.0, 0, 2);
		A.setAsDouble(3.0, 1, 1);
		A.setAsDouble(4.0, 1, 3);

		LinearConstraint c = new LinearConstraint(A,
				new double[] {0.0, 0.0}, new double[] {0.0, 0.0});
		Assertions.assertEquals(2, c.nEq());
		Assertions.assertEquals(0, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 2, 3, 4});
		Matrix Jeq = c.jacEq(x);
		Assertions.assertTrue(Jeq.isSparse(),
				"jacEq should return sparse Matrix when A is sparse");
		Assertions.assertEquals(1.0, Jeq.getAsDouble(0, 0), 1.0e-12);
		Assertions.assertEquals(2.0, Jeq.getAsDouble(0, 2), 1.0e-12);
		Assertions.assertEquals(0.0, Jeq.getAsDouble(0, 1), 1.0e-12);
		Assertions.assertEquals(3.0, Jeq.getAsDouble(1, 1), 1.0e-12);
		Assertions.assertEquals(4.0, Jeq.getAsDouble(1, 3), 1.0e-12);
	}

	@Test
	void sparseAjacIneqAppliesSignFlipsAndStaysSparse() {
		// 1x2 sparse matrix with two-sided bounds — produces both
		// ub-row (sign +1) and lb-row (sign -1) inequalities.
		SparseMatrix A = SparseMatrix.Factory.zeros(1, 2);
		A.setAsDouble(2.0, 0, 0);
		A.setAsDouble(3.0, 0, 1);

		LinearConstraint c = new LinearConstraint(A,
				new double[] {0.0}, new double[] {6.0});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(2, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 1});
		Matrix Jineq = c.jacIneq(x);
		Assertions.assertTrue(Jineq.isSparse(),
				"jacIneq should return sparse Matrix when A is sparse");
		// Row 0 is ub (sign +1): A[0,:] = [2, 3].
		Assertions.assertEquals(2.0, Jineq.getAsDouble(0, 0), 1.0e-12);
		Assertions.assertEquals(3.0, Jineq.getAsDouble(0, 1), 1.0e-12);
		// Row 1 is lb (sign -1): -A[0,:] = [-2, -3].
		Assertions.assertEquals(-2.0, Jineq.getAsDouble(1, 0), 1.0e-12);
		Assertions.assertEquals(-3.0, Jineq.getAsDouble(1, 1), 1.0e-12);
	}

	@Test
	void csrJacobianBuildsDirectlyFromSparseA() {
		// jacEqCSR / jacIneqCSR build CSRMatrix without going through dense
		// when A is sparse — verifying just the entries match.
		SparseMatrix A = SparseMatrix.Factory.zeros(3, 4);
		A.setAsDouble(5.0, 0, 0);
		A.setAsDouble(6.0, 0, 3);
		A.setAsDouble(7.0, 2, 1);
		A.setAsDouble(8.0, 2, 2);

		LinearConstraint c = new LinearConstraint(A,
				new double[] {0, 0, 0}, new double[] {0, 0, 0});
		CSRMatrix csrEq = c.jacEqCSR();
		Assertions.assertEquals(3, csrEq.rows());
		Assertions.assertEquals(4, csrEq.cols());
		Assertions.assertEquals(4, csrEq.nnz());

		double[][] dense = csrEq.toDense();
		Assertions.assertEquals(5.0, dense[0][0], 1.0e-12);
		Assertions.assertEquals(6.0, dense[0][3], 1.0e-12);
		Assertions.assertEquals(7.0, dense[2][1], 1.0e-12);
		Assertions.assertEquals(8.0, dense[2][2], 1.0e-12);
		Assertions.assertEquals(0.0, dense[1][0], 1.0e-12);
		Assertions.assertEquals(0.0, dense[1][1], 1.0e-12);
	}

	@Test
	void hyperplaneRosenbrockWithSparseAConverges() {
		// End-to-end: hyperplane Rosenbrock with x[0] + x[1] = 2, but
		// constraint matrix supplied as a sparse matrix. The orchestrator
		// detects A.isSparse() and routes through the AUGMENTED_SYSTEM
		// factorization in Projections; should still converge to (1, 1).
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
					{-400.0 * a, 200.0}});
		};

		SparseMatrix A = SparseMatrix.Factory.zeros(1, 2);
		A.setAsDouble(1.0, 0, 0);
		A.setAsDouble(1.0, 0, 1);
		LinearConstraint eq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0, eq,
				1000, 1.0e-10, 1.0e-10);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		Assertions.assertTrue(r.fun < 1.0e-12,
				"Sparse-A path should converge to machine precision; got fun=" + r.fun);
	}
}
