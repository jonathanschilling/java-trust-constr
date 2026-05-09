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

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.enums.ProjectionMethod;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Tests that the {@code initial*} tuning knobs and {@code factorizationMethod}
 * parameter on the full-shape {@link MinimizeTrustConstr#minimizeTrustConstr}
 * entry point are honored -- i.e. they reach the inner SQP / IP loop instead
 * of being silently overridden by the hardcoded defaults the convenience
 * overloads use.
 */
class TestTuningKnobs {

	private static final ToDoubleBiFunction<Matrix, Object> rosenFun = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenGrad = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
				200.0 * (b - a * a)});
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenHess = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
				{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
				{-400.0 * a, 200.0}});
	};

	@Test
	void smallInitialTrustRadiusOnEquality() {
		// Hyperplane Rosenbrock with a tiny initial trust radius. Same
		// tolerances and starting point as TestEqualityConstrainedRosenbrock
		// -- converges to (1, 1), but takes more iterations than the default
		// trust radius (1.0) since the algorithm has to grow the radius.
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult tiny = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				null,
				1000, 0, null,
				1.0,                   // initialConstraintPenalty (default)
				1.0e-3,                // initialTrustRadius -- TIGHT
				0.1, 0.1,
				null, false);

		// Same start with the default trust radius for comparison.
		OptimizeResult dflt = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, tiny.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, dflt.x.getAsDouble(0, 0), 1.0e-6);
		// Tight initial trust radius forces extra iterations -- the knob is
		// reaching the SQP loop.
		Assertions.assertTrue(tiny.nIter > dflt.nIter,
				"Tight initial trust radius should require more iterations; "
						+ "tiny=" + tiny.nIter + " dflt=" + dflt.nIter);
	}

	@Test
	void factorizationMethodReachesSqpLoop() {
		// Exercise the QR / SVD / AugmentedSystem options on the equality
		// path. All three should converge to the same answer; we just check
		// that none crashes (i.e., the knob is plumbed through and the
		// alternative factorization paths are wired up).
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});

		for (ProjectionMethod m : new ProjectionMethod[] {
				ProjectionMethod.QR_FACTORIZATION,
				ProjectionMethod.SVD_FACTORIZATION,
				ProjectionMethod.AUGMENTED_SYSTEM}) {
			OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
					rosenFun, x0, null,
					rosenGrad, rosenHess,
					null, null, eq,
					1.0e-10, 1.0e-10, 1.0e-10,
					Optional.empty(),
					null,
					1000, 0, null,
					1.0, 1.0, 0.1, 0.1,
					m, false);
			Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6,
					"factorizationMethod=" + m + " produced unexpected x[0]=" + r.x.getAsDouble(0, 0));
			Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		}
	}

	@Test
	void barrierParameterReachesIpLoop() {
		// IP path: a larger initial barrier parameter forces extra outer
		// barrier-decay iterations before the algorithm reaches the gtol
		// stopping condition.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});

		OptimizeResult dflt = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, ineq,
				1.0e-8, 1.0e-8, 1.0e-8,
				Optional.empty(), null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		OptimizeResult bigBarrier = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, ineq,
				1.0e-8, 1.0e-8, 1.0e-8,
				Optional.empty(), null,
				1000, 0, null,
				1.0, 1.0,
				10.0,                  // initialBarrierParameter -- LARGE
				10.0,                  // initialBarrierTolerance -- LARGE
				null, false);

		Assertions.assertEquals(2.0, dflt.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(2.0, bigBarrier.x.getAsDouble(0, 0), 1.0e-3);
		// Larger initial barrier requires more decay steps to reach gtol.
		Assertions.assertTrue(bigBarrier.nIter > dflt.nIter,
				"Larger initial barrier should require more iterations; "
						+ "big=" + bigBarrier.nIter + " dflt=" + dflt.nIter);
	}

	@Test
	void finiteDifferenceRelStepReachesGradientPath() {
		// Hands-off entry: only objective. The full-shape adapter routes
		// through buildFdGrad with the user's finiteDifferenceRelStep.
		// A coarse step (1e-3) should still converge for a smooth quadratic
		// -- verifying the knob is plumbed through to NumDiff.
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		Matrix coarseStep = Matrix.Factory.linkToArray(new double[] {1.0e-3, 1.0e-3});

		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				(x, args) -> x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
						+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0),
				x0, null,
				null, null,           // grad=null -> FD path; hess=null -> BFGS
				null, null, eq,
				1.0e-6, 1.0e-6, 1.0e-6,
				Optional.empty(),
				null,
				500, 0,
				coarseStep,            // finiteDifferenceRelStep
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-2);
		Assertions.assertTrue(Math.abs(r.fun - 2.0) < 1.0e-2);
	}

	@Test
	void barrierTolSeparateFromGtol() {
		// Active-bound Rosenbrock through the IP path. Loosening barrierTol
		// (the lower bound on barrierParameter at xtol-based termination)
		// independently of gTol should change the iteration count without
		// changing the final answer. Verifies barrierTol is plumbed
		// separately from gTol.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});

		// Both runs use gTol=1e-8. The first uses barrierTol=1e-8 (=gTol);
		// the second uses barrierTol=1e-2 (much looser). The looser
		// barrier-tolerance run should terminate earlier (or at least no
		// later) than the matched-tolerance run.
		OptimizeResult tight = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, ineq,
				1.0e-8, 1.0e-8, 1.0e-8,        // xtol, gtol, barrierTol
				Optional.empty(), null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);
		OptimizeResult loose = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, ineq,
				1.0e-8, 1.0e-8, 1.0e-2,        // barrierTol much looser
				Optional.empty(), null,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(2.0, tight.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(2.0, loose.x.getAsDouble(0, 0), 1.0e-3);
		// The looser barrier tolerance can stop earlier via the
		// "trustRadius < xtol && barrierParameter < barrierTol" branch.
		Assertions.assertTrue(loose.nIter <= tight.nIter,
				"Looser barrierTol should not require more iterations; "
						+ "loose=" + loose.nIter + " tight=" + tight.nIter);
	}

	@Test
	void verbosePrintsIterationLines() {
		// verbose>=1 (or disp=true) should auto-install a printing callback
		// that emits one line per iteration. Capture stdout and check that
		// at least one matching line appears.
		PrintStream origOut = System.out;
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		System.setOut(new PrintStream(captured));

		try {
			LinearConstraint eq = new LinearConstraint(
					Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
					new double[] {2.0}, new double[] {2.0});

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
			OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
					rosenFun, x0, null,
					rosenGrad, rosenHess,
					null, null, eq,
					1.0e-10, 1.0e-10, 1.0e-10,
					Optional.empty(),
					null,                  // callback null -> auto-printer fires
					1000, 1, null,         // verbose=1
					1.0, 1.0, 0.1, 0.1,
					null, false);
			Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		} finally {
			System.setOut(origOut);
		}

		String output = captured.toString();
		// Header should appear once.
		Assertions.assertTrue(output.contains("niter"),
				"Verbose output should contain header; got:\n" + output);
		// At least one iteration line. SQPReport uses 7-char-wide niter
		// column (mirrors scipy), so "1" becomes "      1" (6 spaces + 1).
		Assertions.assertTrue(output.contains("|      1|"),
				"Verbose output should contain iteration 1; got:\n" + output);
	}

	@Test
	void dispBumpsVerboseToOne() {
		// disp=true with verbose=0 should equal verbose=1.
		PrintStream origOut = System.out;
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		System.setOut(new PrintStream(captured));

		try {
			LinearConstraint eq = new LinearConstraint(
					Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
					new double[] {2.0}, new double[] {2.0});

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
			MinimizeTrustConstr.minimizeTrustConstr(
					rosenFun, x0, null,
					rosenGrad, rosenHess,
					null, null, eq,
					1.0e-10, 1.0e-10, 1.0e-10,
					Optional.empty(),
					null,
					1000, 0, null,         // verbose=0 ...
					1.0, 1.0, 0.1, 0.1,
					null, true);           // ... but disp=true
		} finally {
			System.setOut(origOut);
		}

		Assertions.assertTrue(captured.toString().contains("niter"),
				"disp=true should auto-bump verbose to 1");
	}

	@Test
	void verboseSilentWhenZero() {
		// verbose=0 with disp=false produces no output.
		PrintStream origOut = System.out;
		ByteArrayOutputStream captured = new ByteArrayOutputStream();
		System.setOut(new PrintStream(captured));

		try {
			LinearConstraint eq = new LinearConstraint(
					Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
					new double[] {2.0}, new double[] {2.0});

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
			MinimizeTrustConstr.minimizeTrustConstr(
					rosenFun, x0, null,
					rosenGrad, rosenHess,
					null, null, eq,
					1.0e-10, 1.0e-10, 1.0e-10,
					Optional.empty(),
					null,
					1000, 0, null,
					1.0, 1.0, 0.1, 0.1,
					null, false);
		} finally {
			System.setOut(origOut);
		}

		Assertions.assertEquals("", captured.toString(),
				"verbose=0 + disp=false should produce no output");
	}
}
