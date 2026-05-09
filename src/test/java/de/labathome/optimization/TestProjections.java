package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.matrix.MatrixOps;
import de.labathome.trustconstr.Projections;
import de.labathome.trustconstr.enums.ProjectionMethod;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.LinAlg;
import de.labathome.trustconstr.matrix.Matrix;


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
				Matrix x2 = LinAlg.qr(DenseMatrix.copyFromMatrix(A.transpose())).solve(DenseMatrix.copyFromMatrix(z));
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
		// Tolerance is a few ulps to absorb LAPACK rounding-mode noise.
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
				int rowsA = (int) A.getRowCount();
				int colsA = (int) A.getColumnCount();
				Matrix extA = Matrix.Factory.zeros(rowsA + 1, colsA);
				for (int i = 0; i < rowsA; ++i) {
					for (int j = 0; j < colsA; ++j) {
						extA.setAsDouble(A.getAsDouble(i, j), i, j);
					}
				}
				int rowsX = (int) x.getRowCount();
				for (int i = 0; i < rowsX; ++i) {
					extA.setAsDouble(x.getAsDouble(i, 0), rowsA, i);
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
				Matrix x2 = LinAlg.qr(DenseMatrix.copyFromMatrix(A.transpose())).solve(DenseMatrix.copyFromMatrix(z));
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
		for (int i = 0; i < n; ++i) {
			double dVal = D.getAsDouble(i, i);
			if (dVal == 0.0) continue;
			sparseA.setAsDouble(dVal, i, 0*n + i);
			sparseA.setAsDouble(dVal, i, 1*n + i);
			sparseA.setAsDouble(dVal, i, 2*n + i);
			sparseA.setAsDouble(dVal, i, 3*n + i);
		}

		Matrix denseA = DenseMatrix.Factory.copyFromMatrix(sparseA);

		// Fix the RNG seed so the random test vectors are reproducible.
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
		// Note: the original code wrote D1's values into all three blocks (apparent typo
		// preserved: `A.setAsDouble(D1.getAsDouble(...), ...)` was used for D2 and D3 too).
		for (int i = 0; i < 3; ++i) {
			double v = D1.getAsDouble(i, i);
			if (v != 0.0) A.setAsDouble(v, i, i);
		}
		for (int i = 0; i < 3; ++i) {
			double v = D1.getAsDouble(i, i);
			if (v != 0.0) A.setAsDouble(v, i, 3 + i);
		}
		for (int i = 0; i < 3; ++i) {
			double v = D1.getAsDouble(i, i);
			if (v != 0.0) A.setAsDouble(v, i, 6 + i);
		}
		Matrix sparseA = MatrixOps.sparse(A);

		// Fix the RNG seed so the random test vectors are reproducible.
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
				int rowsA = (int) A.getRowCount();
				int colsA = (int) A.getColumnCount();
				Matrix extA = Matrix.Factory.zeros(rowsA + 1, colsA);
				for (int i = 0; i < rowsA; ++i) {
					for (int j = 0; j < colsA; ++j) {
						extA.setAsDouble(A.getAsDouble(i, j), i, j);
					}
				}
				int rowsX = (int) x.getRowCount();
				for (int i = 0; i < rowsX; ++i) {
					extA.setAsDouble(x.getAsDouble(i, 0), rowsA, i);
				}
				Assertions.assertEquals(A.rank(), extA.rank());
			}
		}
	}
}
