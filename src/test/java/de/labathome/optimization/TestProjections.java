package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;
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
		final double tolerance = 1.0e-10;

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
				Matrix z = Matrix.Factory.linkToArray(testPoint);

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

	@Test
	void testIterativeRefinementsSparse() {

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		Matrix sparseA = LinAlg.sparse(A);

		double[][] testPoints = {
				{1, 2, 3, 4, 5, 6, 7, 8},
                {1, 10, 3, 0, 1, 6, 7, 8},
                {1.12, 10, 0, 0, 100000, 6, 0.7, 8},
                {1, 0, 0, 0, 0, 1, 2, 3+1e-10}
		};

		final double orthTol = 1.0e-18;
		final int maxRefine = 100;

		for (ProjectionMethod method: SPARSE_METHODS) {
			LinearOperator[] op = Projections.projections(sparseA, method, orthTol, maxRefine);
			LinearOperator Z = op[0]; // nullspace

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint);

				// Test if x is in the null_space
				Matrix x = Z.apply(z);
				double aTol = 1.0e-13 * x.normInf();
				MinervaAssertions.assertArrayRelAbsEquals(new double[(int) A.getRowCount()], LinAlg.col(A.mtimes(x)), aTol);

				// Test orthogonality
				MinervaAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), 1.0e-13);
			}
		}
	}

	@Test
	void testRowspaceSparse() {
		final double tolerance = 1.0e-15;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		Matrix sparseA = LinAlg.sparse(A);

		double[][] testPoints = {
				{1, 2, 3},
                {1, 10, 3},
                {1.12, 10, 0}
		};

		for (ProjectionMethod method: SPARSE_METHODS) {
			LinearOperator[] op = Projections.projections(sparseA, method);
			LinearOperator Y = op[2]; // rowspace

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint);

				// Test if x is solution of A x = z
				Matrix x = Y.apply(z);
				MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(z), LinAlg.col(A.mtimes(x)), tolerance);

				// Test if x is in the return row space of A
				long n = A.getRowCount();
				Matrix extA = Matrix.Factory.zeros(n+1, A.getColumnCount());
				for (long[] pos: A.allCoordinates()) {
					extA.setAsDouble(A.getAsDouble(pos), pos);
				}
				for (long[] pos: x.allCoordinates()) {
					extA.setAsDouble(x.getAsDouble(pos), n, pos[0]);
				}
				Assertions.assertEquals(A.rank(), extA.rank());
			}
		}
	}

	@Test
	void testNullspaceAndLeastSquaresDense() {

	}

	@Test
	void testCompareDenseAndSparse() {

	}

	@Test
	void testCompareDenseAndSparse2() {

	}

	@Test
	void testIterativeRefinementsDense() {

	}

	@Test
	void testRowspaceDense() {

	}
}
