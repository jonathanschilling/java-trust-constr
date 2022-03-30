package de.labathome.optimization;

import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.NumDiff;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

class TestNumDiff {

	@Test
	void testGroupColumns() {
		Matrix structure = Matrix.Factory.linkToArray(new int[][] {
				{1, 1, 0, 0, 0, 0},
		        {1, 1, 1, 0, 0, 0},
		        {0, 1, 1, 1, 0, 0},
		        {0, 0, 1, 1, 1, 0},
		        {0, 0, 0, 1, 1, 1},
		        {0, 0, 0, 0, 1, 1},
		        {0, 0, 0, 0, 0, 0}
		});

		List<Function<Matrix, Matrix> > transforms = new LinkedList<>();
		transforms.add((Matrix A) -> { return DenseMatrix.Factory.copyFromMatrix(A); });
		transforms.add((Matrix A) -> { return SparseMatrix.Factory.copyFromMatrix(A); });

		for (Function<Matrix, Matrix> transform: transforms) {
			Matrix A = transform.apply(structure);

			int[] order = {0, 1, 2, 3, 4, 5};
			int[] expectedGrouds = {0, 1, 2, 0, 1, 2};
			int[] groups = NumDiff.groupColumns(A, order);
			Assertions.assertArrayEquals(expectedGrouds, groups);

			int[] order2 = {1, 2, 4, 3, 5, 0};
			int[] expectedGrouds2 = {2, 0, 1, 2, 0, 1};
			int[] groups2 = NumDiff.groupColumns(A, order2);
			Assertions.assertArrayEquals(expectedGrouds2, groups2);
		}

		// Test repeatability
		int[] groups1 = NumDiff.groupColumns(structure);
		int[] groups2 = NumDiff.groupColumns(structure);
		Assertions.assertArrayEquals(groups1, groups2);
	}

}
