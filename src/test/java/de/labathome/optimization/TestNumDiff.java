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

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.NumDiff;
import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;


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
		final double tolerance = 1.0e-8;

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
			double epsForMethod = NumDiff.epsForMethod(double.class, double.class, method);
			RelAbsAssertions.assertRelAbsEquals(relativeStepsD.get(method), epsForMethod, tolerance);
		}

		// check another FP size
		final float EPSf = Math.ulp((float) 1.0);
		Map<FiniteDifferenceMethod, Float> relativeStepsF = new HashMap<>();
		relativeStepsF.put(FiniteDifferenceMethod.TWO_POINT, (float) Math.sqrt(EPSf));
		relativeStepsF.put(FiniteDifferenceMethod.COMPLEX_STEP, (float) Math.sqrt(EPSf));
		relativeStepsF.put(FiniteDifferenceMethod.THREE_POINT, (float) Math.pow(EPSf, 1.0/3.0));

		for (FiniteDifferenceMethod method: methods) {
			double epsForMethod = NumDiff.epsForMethod(double.class, float.class, method);
			RelAbsAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);

			epsForMethod = NumDiff.epsForMethod(float.class, double.class, method);
			RelAbsAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);

			epsForMethod = NumDiff.epsForMethod(float.class, float.class, method);
			RelAbsAssertions.assertRelAbsEquals(relativeStepsF.get(method), epsForMethod, tolerance);
		}
	}

	@Test
	void testComputeAbsoluteStep() {
		final double tolerance = 1.0e-15;

		// tests calculation of absolute step from rel_step
		final FiniteDifferenceMethod[] methods = {
				FiniteDifferenceMethod.TWO_POINT,
				FiniteDifferenceMethod.COMPLEX_STEP,
				FiniteDifferenceMethod.THREE_POINT
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0e-5, 0.0, 1.0, 1.0e5});

		final double EPS = Math.ulp(1.0);
		Map<FiniteDifferenceMethod, Double> relativeSteps = new HashMap<>();
		relativeSteps.put(FiniteDifferenceMethod.TWO_POINT, Math.sqrt(EPS));
		relativeSteps.put(FiniteDifferenceMethod.COMPLEX_STEP, Math.sqrt(EPS));
		relativeSteps.put(FiniteDifferenceMethod.THREE_POINT, Math.pow(EPS, 1.0/3.0));

		Matrix f0 = Matrix.Factory.linkToArray(new double[] {1.0});

		for (FiniteDifferenceMethod method: methods) {
			double relStep = relativeSteps.get(method);

			double[] correctSteps = new double[] {
				relStep,
				relStep * 1.0,
				relStep * 1.0,
				relStep * Math.abs(x0.getAsDouble(3, 0))
			};

			Matrix absStep = NumDiff.computeAbsoluteStep(null, x0, f0, method);
			RelAbsAssertions.assertArrayRelAbsEquals(correctSteps, absStep.toColumnArray(), tolerance);

			// signX0[i] = +1 if -x0[i] >= 0 else -1
			double[] signX0Arr = new double[(int) x0.getRowCount()];
			for (int i = 0; i < signX0Arr.length; ++i) {
				signX0Arr[i] = -x0.getAsDouble(i, 0) >= 0.0 ? 1.0 : -1.0;
			}
			Matrix signX0 = DenseMatrix.column(signX0Arr);
			absStep = NumDiff.computeAbsoluteStep(null, x0.times(-1), f0, method);
			RelAbsAssertions.assertArrayRelAbsEquals(correctSteps, absStep.times(signX0).toColumnArray(), tolerance);
		}

		// if a relative step is provided it should be used
		double[] relSteps = {0.1, 1, 10, 100};
		double[] correctSteps = {
				relSteps[0] * x0.getAsDouble(0, 0),
				relativeSteps.get(FiniteDifferenceMethod.TWO_POINT),
				relSteps[2] * 1.0,
				relSteps[3] * Math.abs(x0.getAsDouble(3, 0)),
		};

		Matrix absStep = NumDiff.computeAbsoluteStep(Matrix.Factory.linkToArray(relSteps), x0, f0, FiniteDifferenceMethod.TWO_POINT);
		RelAbsAssertions.assertArrayRelAbsEquals(correctSteps, absStep.toColumnArray(), tolerance);

		double[] signX0Arr = new double[(int) x0.getRowCount()];
		for (int i = 0; i < signX0Arr.length; ++i) {
			signX0Arr[i] = -x0.getAsDouble(i, 0) >= 0.0 ? 1.0 : -1.0;
		}
		Matrix signX0 = DenseMatrix.column(signX0Arr);
		absStep = NumDiff.computeAbsoluteStep(Matrix.Factory.linkToArray(relSteps), x0.times(-1), f0, FiniteDifferenceMethod.TWO_POINT);
		RelAbsAssertions.assertArrayRelAbsEquals(correctSteps, absStep.times(signX0).toColumnArray(), tolerance);
	}
}
