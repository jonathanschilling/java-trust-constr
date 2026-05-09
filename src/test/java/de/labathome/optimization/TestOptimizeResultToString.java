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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.enums.PCGStoppingCondition;
import de.labathome.trustconstr.enums.TrustConstrMethod;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

class TestOptimizeResultToString {

	@Test
	void summarisesKeyFields() {
		OptimizeResult r = new OptimizeResult();
		r.success = true;
		r.status = 1;
		r.method = TrustConstrMethod.EQUALITY_CONSTRAINED_SQP;
		r.x = Matrix.Factory.linkToArray(new double[] {1.0, 2.0, 3.0});
		r.fun = 6.0;
		r.optimality = 1.23e-9;
		r.constraintViolation = 0.0;
		r.nIter = 7;
		r.numFunctionEval = 12;
		r.message = "`gtol` termination condition is satisfied.";
		r.cgStopCond = PCGStoppingCondition.NEGATIVE_CURVATURE;

		String s = r.toString();
		// Must include the load-bearing fields.
		Assertions.assertTrue(s.contains("success=true"), s);
		Assertions.assertTrue(s.contains("status=1"), s);
		Assertions.assertTrue(s.contains("EQUALITY_CONSTRAINED_SQP"), s);
		Assertions.assertTrue(s.contains("x=[1.00000, 2.00000, 3.00000]"), s);
		Assertions.assertTrue(s.contains("fun=6.00000"), s);
		Assertions.assertTrue(s.contains("nIter=7"), s);
		Assertions.assertTrue(s.contains("numFunctionEval=12"), s);
		Assertions.assertTrue(s.contains("`gtol` termination"), s);
	}

	@Test
	void truncatesLongVectors() {
		OptimizeResult r = new OptimizeResult();
		double[] longX = new double[20];
		for (int i = 0; i < 20; ++i) longX[i] = i;
		r.x = Matrix.Factory.linkToArray(longX);

		String s = r.toString();
		// First 6 entries appear, then the remainder is summarised.
		Assertions.assertTrue(s.contains("0.00000, 1.00000"), s);
		Assertions.assertTrue(s.contains("...(14 more)"), s);
	}

	@Test
	void handlesNullX() {
		OptimizeResult r = new OptimizeResult();
		// All defaults; x is null.
		String s = r.toString();
		Assertions.assertTrue(s.contains("x=null"), s);
	}
}
