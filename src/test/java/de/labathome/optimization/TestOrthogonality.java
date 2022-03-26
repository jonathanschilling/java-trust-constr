package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.SparseDoubleMatrix2D;

import de.labathome.LinAlg;
import minerva.tests.junit.MinervaAssertions;

public class TestOrthogonality {

	@Test
	void testDenseMatrix() {
		final double tolerance = 1.0e-9;

		Matrix A = Matrix.Factory.importFromArray(new double[][] {
				{1, 2, 3, 4, 0, 5, 0, 7},
				{0, 8, 7, 0, 1, 5, 9, 0},
				{1, 0, 0, 0, 0, 1, 2, 3}
		});

		double[][] testVectors = {
				{ -1.98931144, -1.56363389,
                    -0.84115584, 2.2864762,
                    5.599141, 0.09286976,
                    1.37040802, -0.28145812 },
				{ 697.92794044, -4091.65114008,
                        -3327.42316335, 836.86906951,
                        99434.98929065, -1285.37653682,
                        -4109.21503806, 2935.29289083 }
		};
		double[] expOrth = {0, 0};

		for (int i=0; i<testVectors.length; ++i) {
			Matrix g = Matrix.Factory.importFromArray(testVectors[i]).transpose();
			double orth = Projections.orthogonality(A, g);
			MinervaAssertions.assertRelAbsEquals(expOrth[i], orth, tolerance);
		}
	}

	@Test
	void testSparseMatrix() {
		final double tolerance = 1.0e-9;

		Matrix A = Matrix.Factory.importFromArray(new double[][] {
			{1, 2, 3, 4, 0, 5, 0, 7},
			{0, 8, 7, 0, 1, 5, 9, 0},
			{1, 0, 0, 0, 0, 1, 2, 3}
		});

		// convert A to sparse matrix
		Matrix sparseA = LinAlg.sparse(A);

		double[][] testVectors = {
				{ -1.98931144, -1.56363389,
                    -0.84115584, 2.2864762,
                    5.599141, 0.09286976,
                    1.37040802, -0.28145812 },
				{ 697.92794044, -4091.65114008,
                        -3327.42316335, 836.86906951,
                        99434.98929065, -1285.37653682,
                        -4109.21503806, 2935.29289083 }
		};
		double[] expOrth = {0, 0};

		for (int i=0; i<testVectors.length; ++i) {
			Matrix g = Matrix.Factory.importFromArray(testVectors[i]).transpose();
			double orth = Projections.orthogonality(sparseA, g);
			MinervaAssertions.assertRelAbsEquals(expOrth[i], orth, tolerance);
		}
	}
}
