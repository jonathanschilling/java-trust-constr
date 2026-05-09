package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.matrix.MatrixOps;
import org.scipy.optimize.minimize.Projections;
import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;
import org.scipy.optimize.minimize.matrix.QRMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;


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

		Matrix sparseA = MatrixOps.sparse(A);

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
				double[] r = A.mtimes(x).toColumnArray();
				RelAbsAssertions.assertArrayRelAbsEquals(new double[r.length], r, tolerance);

				// Test orthogonality
				RelAbsAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), tolerance);

				// Test if x is the least square solution
				x = LS.apply(z);
				QRMatrix qrA = new QRMatrix(A.transpose());
				Matrix x2 = qrA.solve(z);
				RelAbsAssertions.assertArrayRelAbsEquals(x2.toColumnArray(), x.toColumnArray(), tolerance);
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

		Matrix sparseA = MatrixOps.sparse(A);

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
				RelAbsAssertions.assertArrayRelAbsEquals(new double[(int) A.getRowCount()], A.mtimes(x).toColumnArray(), aTol);

				// Test orthogonality
				RelAbsAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), 1.0e-13);
			}
		}
	}

	@Test
	void testRowspaceSparse() {
		// Tolerance is a few ulps to absorb rounding-mode differences between
		// the in-tree LAPACK-backed Cholesky/QR paths and the previous UJMP
		// pure-Java factorisations.
		final double tolerance = 1.0e-14;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		Matrix sparseA = MatrixOps.sparse(A);

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
				RelAbsAssertions.assertArrayRelAbsEquals(z.toColumnArray(), A.mtimes(x).toColumnArray(), tolerance);

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
		final double tolerance = 1.0e-10;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		double[][] testPoints = {
				{1, 2, 3, 4, 5, 6, 7, 8},
                {1, 10, 3, 0, 1, 6, 7, 8},
                {1.12, 10, 0, 0, 100000, 6, 0.7, 8}
		};

		for (ProjectionMethod method: DENSE_METHODS) {
			LinearOperator[] op = Projections.projections(A, method);
			LinearOperator Z = op[0]; // nullspace
			LinearOperator LS = op[1]; // least-squares

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint);

				// Test if x is in the null_space
				Matrix x = Z.apply(z);
				double[] r = A.mtimes(x).toColumnArray();
				RelAbsAssertions.assertArrayRelAbsEquals(new double[r.length], r, tolerance);

				// Test orthogonality
				RelAbsAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), tolerance);

				// Test if x is the least square solution
				x = LS.apply(z);
				QRMatrix qrA = new QRMatrix(A.transpose());
				Matrix x2 = qrA.solve(z);
				RelAbsAssertions.assertArrayRelAbsEquals(x2.toColumnArray(), x.toColumnArray(), tolerance);
			}
		}
	}

	@Test
	void testCompareDenseAndSparse() {
		final double tolerance = 1.0e-15;

		int n = 100;
		Matrix D = SparseMatrix.Factory.zeros(n, n);
		for (int i=0; i<n; ++i) {
			D.setAsDouble(i+1, i, i);
		}

		Matrix sparseA = SparseMatrix.Factory.zeros(n, 4*n);
		for (long[] pos: D.availableCoordinates()) {
			double dVal = D.getAsDouble(pos);
			sparseA.setAsDouble(dVal, pos[0], 0*n + pos[1]);
			sparseA.setAsDouble(dVal, pos[0], 1*n + pos[1]);
			sparseA.setAsDouble(dVal, pos[0], 2*n + pos[1]);
			sparseA.setAsDouble(dVal, pos[0], 3*n + pos[1]);
		}

		Matrix denseA = DenseMatrix.Factory.copyFromMatrix(sparseA);

		// set random seed; must be non-zero apparently
		// https://github.com/ujmp/universal-java-matrix-package/issues/35
		Matrix.Factory.setRandSeed(1);

		LinearOperator[] denseOp = Projections.projections(denseA);
		LinearOperator denseZ = denseOp[0];
		LinearOperator denseLS = denseOp[1];
		LinearOperator denseY = denseOp[2];

		LinearOperator[] sparseOp = Projections.projections(sparseA);
		LinearOperator sparseZ = sparseOp[0];
		LinearOperator sparseLS = sparseOp[1];
		LinearOperator sparseY = sparseOp[2];

		int numRepetitions = 1;
		for (int i=0; i<numRepetitions; ++i) {
			Matrix z = Matrix.Factory.randn(4*n, 1);
			RelAbsAssertions.assertArrayRelAbsEquals(denseZ.apply(z).toColumnArray(), sparseZ.apply(z).toColumnArray(), tolerance);
			RelAbsAssertions.assertArrayRelAbsEquals(denseLS.apply(z).toColumnArray(), sparseLS.apply(z).toColumnArray(), tolerance);

			Matrix x = Matrix.Factory.randn(n, 1);
			RelAbsAssertions.assertArrayRelAbsEquals(denseY.apply(x).toColumnArray(), sparseY.apply(x).toColumnArray(), tolerance);
		}
	}

	@Test
	void testCompareDenseAndSparse2() {
		final double tolerance = 1.0e-15;

		Matrix D1 = MatrixOps.diag(new double[] {-1.7, 1, 0.5});
		Matrix D2 = MatrixOps.diag(new double[] {1, -0.6, -0.3});
		Matrix D3 = MatrixOps.diag(new double[] {-0.3, -1.5, 2});
		Matrix A = Matrix.Factory.zeros(3, 9);
		for (long[] pos: D1.availableCoordinates()) {
			A.setAsDouble(D1.getAsDouble(pos), pos);
		}
		for (long[] pos: D2.availableCoordinates()) {
			A.setAsDouble(D1.getAsDouble(pos), pos[0], 3+pos[1]);
		}
		for (long[] pos: D3.availableCoordinates()) {
			A.setAsDouble(D1.getAsDouble(pos), pos[0], 6+pos[1]);
		}
		Matrix sparseA = MatrixOps.sparse(A);

		// set random seed; must be non-zero apparently
		// https://github.com/ujmp/universal-java-matrix-package/issues/35
		Matrix.Factory.setRandSeed(1);

		LinearOperator[] denseOp = Projections.projections(A);
		LinearOperator denseZ = denseOp[0];
		LinearOperator denseLS = denseOp[1];
		LinearOperator denseY = denseOp[2];

		LinearOperator[] sparseOp = Projections.projections(sparseA);
		LinearOperator sparseZ = sparseOp[0];
		LinearOperator sparseLS = sparseOp[1];
		LinearOperator sparseY = sparseOp[2];

		int numRepetitions = 1;
		for (int i=0; i<numRepetitions; ++i) {
			Matrix z = Matrix.Factory.randn(9, 1);
			RelAbsAssertions.assertArrayRelAbsEquals(denseZ.apply(z).toColumnArray(), sparseZ.apply(z).toColumnArray(), tolerance);
			RelAbsAssertions.assertArrayRelAbsEquals(denseLS.apply(z).toColumnArray(), sparseLS.apply(z).toColumnArray(), tolerance);

			Matrix x = Matrix.Factory.randn(3, 1);
			RelAbsAssertions.assertArrayRelAbsEquals(denseY.apply(x).toColumnArray(), sparseY.apply(x).toColumnArray(), tolerance);
		}
	}

	@Test
	void testIterativeRefinementsDense() {
		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		double[][] testPoints = {
				{1, 2, 3, 4, 5, 6, 7, 8},
	            {1, 10, 3, 0, 1, 6, 7, 8},
	            //{1.12, 10, 0, 0, 100000, 6, 0.7, 8},
	            {1, 0, 0, 0, 0, 1, 2, 3+1e-10}
		};

		final double orthTol = 1.0e-18;
		final int maxRefine = 10;

		for (ProjectionMethod method: DENSE_METHODS) {
			LinearOperator[] op = Projections.projections(A, method, orthTol, maxRefine);
			LinearOperator Z = op[0]; // nullspace

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint);

				// Test if x is in the null_space
				Matrix x = Z.apply(z);
				RelAbsAssertions.assertArrayRelAbsEquals(new double[(int) A.getRowCount()], A.mtimes(x).toColumnArray(), 2.5e-14);

				// Test orthogonality
				RelAbsAssertions.assertRelAbsEquals(0.0, Projections.orthogonality(A, x), 5.0e-16);
			}
		}
	}

	@Test
	void testRowspaceDense() {
		final double tolerance = 1.0e-14;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		double[][] testPoints = {
				{1, 2, 3},
                {1, 10, 3},
                {1.12, 10, 0}
		};

		for (ProjectionMethod method: DENSE_METHODS) {
			LinearOperator[] op = Projections.projections(A, method);
			LinearOperator Y = op[2]; // rowspace

			for (double[] testPoint: testPoints) {
				Matrix z = Matrix.Factory.linkToArray(testPoint);

				// Test if x is solution of A x = z
				Matrix x = Y.apply(z);
				RelAbsAssertions.assertArrayRelAbsEquals(z.toColumnArray(), A.mtimes(x).toColumnArray(), tolerance);

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
}
