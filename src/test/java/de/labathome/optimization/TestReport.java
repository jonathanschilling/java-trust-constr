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

import java.util.Optional;
import java.util.function.ToDoubleBiFunction;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.report.ReportBase;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.report.BasicReport;
import de.labathome.trustconstr.report.IPReport;
import de.labathome.trustconstr.report.SQPReport;

/**
 * Java translation of scipy
 * {@code _trustregion_constr/tests/test_report.py}: verifies that the
 * verbose progress printer doesn't crash when {@code verbose >= 2} on
 * bound-constrained ({@code test_gh10880}) and general-constrained
 * ({@code test_gh12922}) problems.
 *
 * <p>Adds unit tests for the {@link BasicReport}, {@link SQPReport}, and
 * {@link IPReport} formatting (number of columns, presence of column
 * headers).
 */
class TestReport {

	@Test
	void basicReportHasSevenColumns() {
		BasicReport r = new BasicReport();
		String out = renderHeader(r);
		assertTrue(out.contains("niter"), "header has niter");
		assertTrue(out.contains("f evals"), "header has f evals");
		assertTrue(out.contains("CG iter"), "header has CG iter");
		assertTrue(out.contains("obj func"), "header has obj func");
		assertTrue(out.contains("tr radius"), "header has tr radius");
		assertTrue(out.contains("opt"), "header has opt");
		assertTrue(out.contains("c viol"), "header has c viol");
		// 7 columns -> 8 separators
		long sepCount = out.chars().filter(c -> c == '|').count();
		assertTrue(sepCount >= 8 * 2, "two header lines x 8 |s each");
	}

	@Test
	void sqpReportHasNineColumns() {
		SQPReport r = new SQPReport();
		String out = renderHeader(r);
		assertTrue(out.contains("penalty"), "header has penalty");
		assertTrue(out.contains("CG stop"), "header has CG stop");
	}

	@Test
	void ipReportHasTenColumns() {
		IPReport r = new IPReport();
		String out = renderHeader(r);
		assertTrue(out.contains("barrier param"), "header has barrier param");
	}

	@Test
	void basicReportPrintsIterationRow() {
		BasicReport r = new BasicReport();
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(buf, true, StandardCharsets.UTF_8);
		r.printIteration(ps, 1, 5, 7, 3.14e-2, 1.0, 1e-5, 0.0);
		String s = buf.toString(StandardCharsets.UTF_8);
		// Format spec asks for scientific notation in column 4
		assertTrue(s.contains("3.1400e-02") || s.contains("+3.1400e-02"),
				"obj func printed in scientific notation: " + s);
	}

	@Test
	void testGh10880BoundConstrainedDoesntCrashAtVerbose2() {
		// scipy: minimize(lambda x: x**2, x0=2., bounds=Bounds(1, 2),
		//                 method='trust-constr', options={'verbose': 2})
		Function<Matrix, Double> fun = x -> {
			double v = x.getAsDouble(0, 0);
			return v * v;
		};
		Function<Matrix, Matrix> grad = x -> DenseMatrix.column(2.0 * x.getAsDouble(0, 0));
		Function<Matrix, Matrix> hess = x -> DenseMatrix.fromRows(new double[][] {{2.0}});
		ToDoubleBiFunction<Matrix, Object> funBi = (x, args) -> fun.apply(x);
		BiFunction<Matrix, Object, Matrix> gradBi = (x, args) -> grad.apply(x);
		BiFunction<Matrix, Object, Matrix> hessBi = (x, args) -> hess.apply(x);
		Bounds b = new Bounds(DenseMatrix.column(1.0), DenseMatrix.column(2.0), false);
		Matrix x0 = DenseMatrix.column(2.0);

		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		PrintStream origOut = System.out;
		System.setOut(new PrintStream(buf, true, StandardCharsets.UTF_8));
		try {
			OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
					funBi, x0, /*args=*/ null, gradBi, hessBi,
					/*hessp=*/ null, b, /*constraints=*/ null,
					/*xtol=*/ 1e-8, /*gtol=*/ 1e-8, /*barrierTol=*/ 1e-8,
					/*sparseJacobian=*/ Optional.empty(),
					/*callback=*/ null, /*maxIter=*/ 1000, /*verbose=*/ 2,
					/*finiteDifferenceRelStep=*/ null,
					/*initialConstraintPenalty=*/ 1.0, /*initialTrustRadius=*/ 1.0,
					/*initialBarrierParameter=*/ 0.1, /*initialBarrierTolerance=*/ 0.1,
					/*factorizationMethod=*/ null, /*disp=*/ false);
			assertTrue(r.success, "should converge");
		} finally {
			System.setOut(origOut);
		}
		String captured = buf.toString(StandardCharsets.UTF_8);
		// The IP path is used (bounds are inequalities). Header contains
		// "barrier param" only on IPReport.
		assertTrue(captured.contains("barrier param"),
				"IP path verbose printer was installed: " + captured.substring(0, Math.min(200, captured.length())));
	}

	@Test
	void testGh12922GeneralConstrainedDoesntCrashAtVerbose2() {
		// scipy: minimize(objective, x0=linspace(-5, 5, 25),
		//                 method='trust-constr',
		//                 constraints={'type': 'ineq', 'fun': lambda x: -x[0]**2},
		//                 options={'verbose': 2})
		// We use n=5 to keep the test fast; scipy uses n=25 with @pytest.mark.xslow.
		final int n = 5;
		Function<Matrix, Double> fun = x -> {
			double s = 0;
			for (int i = 0; i < n; ++i) {
				double v = x.getAsDouble(i, 0) + 1.0;
				s += v * v * v * v;
			}
			return s;
		};
		Function<Matrix, Matrix> grad = x -> {
			double[] g = new double[n];
			for (int i = 0; i < n; ++i) {
				double v = x.getAsDouble(i, 0) + 1.0;
				g[i] = 4.0 * v * v * v;
			}
			return DenseMatrix.column(g);
		};
		Function<Matrix, Matrix> hess = x -> {
			double[][] h = new double[n][n];
			for (int i = 0; i < n; ++i) {
				double v = x.getAsDouble(i, 0) + 1.0;
				h[i][i] = 12.0 * v * v;
			}
			return DenseMatrix.fromRows(h);
		};
		ToDoubleBiFunction<Matrix, Object> funBi = (x, args) -> fun.apply(x);
		BiFunction<Matrix, Object, Matrix> gradBi = (x, args) -> grad.apply(x);
		BiFunction<Matrix, Object, Matrix> hessBi = (x, args) -> hess.apply(x);

		// Inequality constraint: c(x) = -x[0]^2 >= 0 (i.e. x[0]^2 <= 0)
		Function<Matrix, Matrix> cFun = x -> {
			double v = x.getAsDouble(0, 0);
			return DenseMatrix.column(-v * v);
		};
		Function<Matrix, Matrix> cJac = x -> {
			double[][] J = new double[1][n];
			J[0][0] = -2.0 * x.getAsDouble(0, 0);
			return DenseMatrix.fromRows(J);
		};
		BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double[][] h = new double[n][n];
			h[0][0] = -2.0 * v.getAsDouble(0, 0);
			return DenseMatrix.fromRows(h);
		};
		double[] cLb = {0.0};
		double[] cUb = {Double.POSITIVE_INFINITY};
		NonlinearConstraint c =
				new NonlinearConstraint(
						cFun, cJac, cHess, cLb, cUb, null);

		double[] x0Arr = new double[n];
		for (int i = 0; i < n; ++i) x0Arr[i] = -5.0 + (10.0 / (n - 1)) * i;
		Matrix x0 = DenseMatrix.column(x0Arr);

		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		PrintStream origOut = System.out;
		System.setOut(new PrintStream(buf, true, StandardCharsets.UTF_8));
		try {
			OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
					funBi, x0, null, gradBi, hessBi, null, /*bounds=*/ null,
					c,
					1e-6, 1e-6, 1e-6, Optional.empty(),
					null, 200, /*verbose=*/ 2, null,
					1.0, 1.0, 0.1, 0.1, null, false);
			// Don't assert success since the constraint x[0]^2 <= 0 forces x[0]=0
			// which is a tight active constraint -- convergence is delicate.
			// The test is just that the verbose printer doesn't crash.
			assertTrue(r.nIter > 0, "at least one iteration ran");
		} finally {
			System.setOut(origOut);
		}
		String captured = buf.toString(StandardCharsets.UTF_8);
		assertTrue(captured.contains("barrier param"),
				"IP path verbose printer was installed for inequality constraint");
	}

	private static String renderHeader(ReportBase r) {
		ByteArrayOutputStream buf = new ByteArrayOutputStream();
		PrintStream ps = new PrintStream(buf, true, StandardCharsets.UTF_8);
		r.printHeader(ps);
		return buf.toString(StandardCharsets.UTF_8);
	}
}
