package de.labathome.optimization;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.NumDiff;
import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

import minerva.tests.junit.MinervaAssertions;

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

	@Test
	void testCorrectFpEps() {
		final double tolerance = 1.0e-15;

		final FiniteDifferenceMethod[] methods = {
				FiniteDifferenceMethod.TWO_POINT,
				FiniteDifferenceMethod.COMPLEX_STEP,
				FiniteDifferenceMethod.THREE_POINT
		};

		// check that relative step size is correct for FP size
		final double EPSd = Math.ulp(1.0);
		Map<FiniteDifferenceMethod, Double> relativeStepsD = new HashMap<>();
		relativeStepsD.put(FiniteDifferenceMethod.TWO_POINT, Math.sqrt(EPSd));
		relativeStepsD.put(FiniteDifferenceMethod.COMPLEX_STEP, Math.sqrt(EPSd));
		relativeStepsD.put(FiniteDifferenceMethod.THREE_POINT, Math.pow(EPSd, 1.0/3.0));

		for (FiniteDifferenceMethod method: methods) {
			double epsForMethod = NumDiff.epsForMethod(double.class, double.class, method).doubleValue();
			MinervaAssertions.assertRelAbsEquals(relativeStepsD.get(method), epsForMethod, tolerance);
		}

		// check another FP size
		final float EPSf = Math.ulp((float) 1.0);
		Map<FiniteDifferenceMethod, Float> relativeStepsF = new HashMap<>();
		relativeStepsF.put(FiniteDifferenceMethod.TWO_POINT, (float) Math.sqrt(EPSf));
		relativeStepsF.put(FiniteDifferenceMethod.COMPLEX_STEP, (float) Math.sqrt(EPSf));
		relativeStepsF.put(FiniteDifferenceMethod.THREE_POINT, (float) Math.pow(EPSf, 1.0/3.0));

		for (FiniteDifferenceMethod method: methods) {
			double epsForMethod = NumDiff.epsForMethod(double.class, float.class, method).doubleValue();
			MinervaAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);

			epsForMethod = NumDiff.epsForMethod(float.class, double.class, method).doubleValue();
			MinervaAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);

			epsForMethod = NumDiff.epsForMethod(float.class, float.class, method).doubleValue();
			MinervaAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);
		}
	}

}
