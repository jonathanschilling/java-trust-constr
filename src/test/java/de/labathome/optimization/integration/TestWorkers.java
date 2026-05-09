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
package de.labathome.optimization.integration;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.OptimizeResult;

import de.labathome.optimization.RelAbsAssertions;

/**
 * Phase 9: strict parity for the {@code workers} parameter on
 * {@code _minimize_trustregion_constr}. When non-{@code null}, the executor
 * dispatches FD-gradient column evaluations in parallel; analytic gradients
 * are unaffected.
 */
class TestWorkers {

	@Test
	void workersOverloadProducesSameResultAsSerial() {
		// 2D quadratic. With analytic grad omitted, the FD-grad path is used,
		// and workers parallelises it.
		ToDoubleBiFunction<Matrix, Object> fun = (x, args) -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (a - 1.0) * (a - 1.0) + (b + 0.5) * (b + 0.5);
		};
		Matrix x0 = DenseMatrix.column(0.0, 0.0);

		// Serial.
		OptimizeResult rSerial = MinimizeTrustConstr.minimizeTrustConstr(
				fun, x0, null, null, null, null, null, null,
				1e-8, 1e-8, 1e-8, Optional.empty(),
				null, 200, 0, null,
				1.0, 1.0, 0.1, 0.1, null, false);

		// Parallel (executor non-null).
		ExecutorService pool = Executors.newFixedThreadPool(2);
		try {
			OptimizeResult rParallel = MinimizeTrustConstr.minimizeTrustConstr(
					fun, x0, null, null, null, null, null, null,
					1e-8, 1e-8, 1e-8, Optional.empty(),
					null, 200, 0, null,
					1.0, 1.0, 0.1, 0.1, null, false,
					pool);
			assertTrue(rParallel.success, "parallel path converges");
			RelAbsAssertions.assertRelAbsEquals(
					rSerial.x.getAsDouble(0, 0),
					rParallel.x.getAsDouble(0, 0), 1e-6, "x[0]");
			RelAbsAssertions.assertRelAbsEquals(
					rSerial.x.getAsDouble(1, 0),
					rParallel.x.getAsDouble(1, 0), 1e-6, "x[1]");
		} finally {
			pool.shutdown();
		}
	}

	@Test
	void workersNullEqualsSerialOverload() {
		// Passing workers=null to the new overload must equal the existing
		// no-workers overload byte-for-byte (identical algorithm, same
		// thread-local cleanup behavior).
		ToDoubleBiFunction<Matrix, Object> fun = (x, args) -> {
			double a = x.getAsDouble(0, 0);
			return a * a;
		};
		BiFunction<Matrix, Object, Matrix> grad = (x, args) -> DenseMatrix.column(
				2.0 * x.getAsDouble(0, 0));
		BiFunction<Matrix, Object, Matrix> hess = (x, args) -> DenseMatrix.fromRows(
				new double[][] {{2.0}});
		Matrix x0 = DenseMatrix.column(1.0);

		OptimizeResult rA = MinimizeTrustConstr.minimizeTrustConstr(
				fun, x0, null, grad, hess, null, null, null,
				1e-8, 1e-8, 1e-8, Optional.empty(),
				null, 200, 0, null,
				1.0, 1.0, 0.1, 0.1, null, false);

		OptimizeResult rB = MinimizeTrustConstr.minimizeTrustConstr(
				fun, x0, null, grad, hess, null, null, null,
				1e-8, 1e-8, 1e-8, Optional.empty(),
				null, 200, 0, null,
				1.0, 1.0, 0.1, 0.1, null, false,
				/*workers=*/ null);

		assertNotNull(rA);
		assertNotNull(rB);
		// Same x, fun, status.
		RelAbsAssertions.assertRelAbsEquals(
				rA.x.getAsDouble(0, 0), rB.x.getAsDouble(0, 0), 1e-12);
		RelAbsAssertions.assertRelAbsEquals(rA.fun, rB.fun, 1e-12);
		assertTrue(rA.status == rB.status);
	}
}
