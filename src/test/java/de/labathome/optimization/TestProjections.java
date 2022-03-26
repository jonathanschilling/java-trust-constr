package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;
import org.ujmp.core.doublematrix.calculation.general.decomposition.QR.QRMatrix;

import de.labathome.LinAlg;
import de.labathome.LinearOperator;
import minerva.tests.junit.MinervaAssertions;

class TestProjections {

	static final ProjectionMethod[] SPARSE_METHODS = {
			ProjectionMethod.NORMAL_EQUATION,
			ProjectionMethod.AUGMENTED_SYSTEM
	};

	static final ProjectionMethod[] DENSE_METHODS = {
			ProjectionMethod.QR_FACTORIZATION,
			ProjectionMethod.SVD_FACTORIZATION
	};

	@Test
	void testNullspaceAndLeastSquaresSparse() {
		final double tolerance = 1.0e-15;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		Matrix sparseA = LinAlg.sparse(A);

		double[][] testPoints = {
				{1, 2, 3, 4, 5, 6, 7, 8},
                {1, 10, 3, 0, 1, 6, 7, 8},
                {1.12, 10, 0, 0, 100000, 6, 0.7, 8}
		};

		for (ProjectionMethod method: SPARSE_METHODS) {
			LinearOperator[] op = Projections.projections(sparseA, method);
			LinearOperator Z = op[0]; // nullspace
			LinearOperator LS = op[1]; // least-squares

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint).transpose();

				// Test if x is in the null_space
				Matrix x = Z.apply(z);
				Matrix residual = A.mtimes(x);
				double[] r = LinAlg.col(residual);
				MinervaAssertions.assertArrayRelAbsEquals(new double[r.length], r, tolerance);

				// Test orthogonality
				MinervaAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), tolerance);

				// Test if x is the least square solution
				x = LS.apply(z);
				QRMatrix qrA = new QRMatrix(A.transpose());
				Matrix x2 = qrA.solve(z);
				MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(x2), LinAlg.col(x), tolerance);
			}
		}
	}

	void testIterativeRefinementsSparse() {






	}
}
