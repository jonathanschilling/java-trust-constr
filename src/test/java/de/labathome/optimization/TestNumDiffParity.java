package de.labathome.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.NumDiff;
import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.ApproxDerivativeResult;
import de.labathome.trustconstr.records.FiniteDifferenceBounds;
import de.labathome.trustconstr.records.FiniteDifferenceOptions;

/**
 * Phase 4 strict-parity tests against scipy's
 * {@code approx_derivative(..., as_linear_operator=True)},
 * {@code approx_derivative(..., workers=N)}, and
 * {@code approx_derivative(..., full_output=True)} APIs.
 */
class TestNumDiffParity {

	private static final double TOL = 1e-6;

	private static FiniteDifferenceOptions defaultOptions(int n) {
		return new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.TWO_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(n))
				.build();
	}

	@Test
	void asLinearOperatorReturnsLinearOperatorOfFD() {
		// f(x) = (x[0]^2, x[0] + x[1]) at x0 = (1, 2)
		// J = [[2*x[0], 0], [1, 1]] = [[2, 0], [1, 1]]
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(1, 0));
		Matrix x0 = DenseMatrix.column(1.0, 2.0);
		FiniteDifferenceOptions opts = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.TWO_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(2))
				.build();
		LinearOperator op = NumDiff.approxDerivativeAsLinearOperator(fun, x0, null, opts, null);
		assertNotNull(op);

		// Apply op to e_0 = (1, 0) -- should give the first column of J = (2, 1).
		Matrix e0 = DenseMatrix.column(1.0, 0.0);
		Matrix Je0 = op.apply(e0);
		// asLinearOperator's matvec involves a directional FD step, so the
		// result is approximate. Tolerance must be loose enough to allow for
		// FD noise.
		RelAbsAssertions.assertRelAbsEquals(2.0, Je0.getAsDouble(0, 0), TOL, "Je0[0]");
		RelAbsAssertions.assertRelAbsEquals(1.0, Je0.getAsDouble(1, 0), TOL, "Je0[1]");
	}

	@Test
	void workersGivesSameResultAsSerial() {
		// f(x) = (x[0]*x[1], x[0] + x[2]^2) at x0 = (1, 2, 3)
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(1, 0),
				x.getAsDouble(0, 0) + x.getAsDouble(2, 0) * x.getAsDouble(2, 0));
		Matrix x0 = DenseMatrix.column(1.0, 2.0, 3.0);
		FiniteDifferenceOptions opts = defaultOptions(3);

		Matrix Jserial = NumDiff.approxDerivative(fun, x0, null, opts, null);

		ExecutorService pool = Executors.newFixedThreadPool(3);
		try {
			Matrix Jparallel = NumDiff.approxDerivativeWithWorkers(fun, x0, null, opts, null, pool);
			assertEquals(Jserial.getRowCount(), Jparallel.getRowCount());
			assertEquals(Jserial.getColumnCount(), Jparallel.getColumnCount());
			for (long i = 0; i < Jserial.getRowCount(); ++i) {
				for (long j = 0; j < Jserial.getColumnCount(); ++j) {
					RelAbsAssertions.assertRelAbsEquals(
							Jserial.getAsDouble(i, j),
							Jparallel.getAsDouble(i, j),
							1e-12, "[" + i + "," + j + "]");
				}
			}
		} finally {
			pool.shutdown();
		}
	}

	@Test
	void workersWithThreePointMatchesSerial() {
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				Math.sin(x.getAsDouble(0, 0)) + x.getAsDouble(1, 0),
				x.getAsDouble(0, 0) * x.getAsDouble(1, 0));
		Matrix x0 = DenseMatrix.column(0.5, 1.5);
		FiniteDifferenceOptions opts = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
				.method(FiniteDifferenceMethod.THREE_POINT)
				.bounds(FiniteDifferenceBounds.unbounded(2))
				.build();

		Matrix Jserial = NumDiff.approxDerivative(fun, x0, null, opts, null);

		ExecutorService pool = Executors.newFixedThreadPool(2);
		try {
			Matrix Jparallel = NumDiff.approxDerivativeWithWorkers(fun, x0, null, opts, null, pool);
			for (long i = 0; i < Jserial.getRowCount(); ++i) {
				for (long j = 0; j < Jserial.getColumnCount(); ++j) {
					RelAbsAssertions.assertRelAbsEquals(
							Jserial.getAsDouble(i, j),
							Jparallel.getAsDouble(i, j),
							1e-12, "[" + i + "," + j + "]");
				}
			}
		} finally {
			pool.shutdown();
		}
	}

	@Test
	void workersFallsBackToSerialWhenNullExecutor() {
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0));
		Matrix x0 = DenseMatrix.column(1.0);
		FiniteDifferenceOptions opts = defaultOptions(1);

		Matrix J = NumDiff.approxDerivativeWithWorkers(fun, x0, null, opts, null, null);
		// J = [2.0]
		assertEquals(1, J.getRowCount());
		assertEquals(1, J.getColumnCount());
		RelAbsAssertions.assertRelAbsEquals(2.0, J.getAsDouble(0, 0), TOL);
	}

	@Test
	void fullOutputReportsNfev() {
		// f(x) = (x[0]^2, x[1]) at x0 = (1, 2). Two-point FD: 1 eval at x0 +
		// 2 evals (one per input dim). nfev should report 3.
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0),
				x.getAsDouble(1, 0));
		Matrix x0 = DenseMatrix.column(1.0, 2.0);
		FiniteDifferenceOptions opts = defaultOptions(2);

		ApproxDerivativeResult r = NumDiff.approxDerivativeFullOutput(fun, x0, null, opts, null);
		assertEquals(2, r.jacobian().getRowCount());
		assertEquals(2, r.jacobian().getColumnCount());
		Map<String, Object> info = r.info();
		assertTrue(info.containsKey("nfev"), "info has nfev");
		assertTrue(info.containsKey("method"), "info has method");
		// 1 (x0 baseline) + 2 (one per input column) = 3 evals for TWO_POINT
		Integer nfev = (Integer) info.get("nfev");
		assertEquals(3, nfev.intValue(), "nfev = baseline + n forward steps");
	}

	@Test
	void fullOutputUsesCachedF0WhenProvided() {
		BiFunction<Matrix, Object, Matrix> fun = (x, a) -> DenseMatrix.column(
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0));
		Matrix x0 = DenseMatrix.column(1.0);
		Matrix f0 = DenseMatrix.column(1.0);
		FiniteDifferenceOptions opts = defaultOptions(1);

		ApproxDerivativeResult r = NumDiff.approxDerivativeFullOutput(fun, x0, f0, opts, null);
		// With f0 provided, only 1 eval (the perturbed step) is needed for n=1.
		Integer nfev = (Integer) r.info().get("nfev");
		assertEquals(1, nfev.intValue(), "nfev skips baseline eval when f0 supplied");
	}
}
