package org.scipy.optimize.minimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.records.AdjustedDifferencingScheme;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
import org.scipy.optimize.minimize.records.Sparsity;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;

/**
 * Finite-difference Jacobians and gradients (mirrors scipy
 * {@code _numdiff.approx_derivative}). Provides:
 * <ul>
 *   <li>{@link #approxDerivative} -- vector-valued FD with optional sparsity hint;</li>
 *   <li>{@link #adjustSchemeToBounds} -- bound-aware step-direction selection;</li>
 *   <li>{@link #computeAbsoluteStep} / {@link #epsForMethod} -- step-size helpers;</li>
 *   <li>{@link #groupColumns} -- Curtis-Powell-Reid column grouping for sparse FD.</li>
 * </ul>
 */
public class NumDiff {

	/**
	 * Group columns of a 2-D matrix for sparse finite differencing.
	 * Two columns are in the same group if in each row at least one of them
	 * has zero. A greedy sequential algorithm constructs the groups
	 * (Curtis-Powell-Reid, 1974).
	 *
	 * @param A {@code m x n} sparsity pattern (non-zero entries mark
	 *          structural non-zeros of the Jacobian).
	 * @return length-{@code n} array of group indices in {@code [0, n_groups)}.
	 */
	public static int[] groupColumns(Matrix A) {
		return groupColumns(A, new int[] {0});
	}

	/**
	 * Group columns of a 2-D matrix for sparse FD with a user-supplied
	 * permutation that controls greedy-grouping order.
	 *
	 * @param A     {@code m x n} sparsity pattern
	 * @param order length-{@code n} permutation; if {@code null} or length {@code &le; 1},
	 *              a reproducible random permutation seeded by {@code order[0]}
	 *              (defaulting to {@code 0}) is used
	 * @return length-{@code n} array of group indices
	 */
	public static int[] groupColumns(Matrix A, int[] order) {
		if (A.getSize().length != 2) {
			throw new RuntimeException("`A` must be 2-dimensional.");
		}

		int m = (int) A.getRowCount();
		int n = (int) A.getColumnCount();

		// Get random, but reproducible order if no order is given.
		if (order == null || order.length <= 1) {
			Random rnd = new Random((order == null || order.length == 0) ? 0 : order[0]);
			List<Integer> indices = new ArrayList<>(n);
			for (int i = 0; i < n; ++i) indices.add(i);
			Collections.shuffle(indices, rnd);
			order = indices.stream().mapToInt(Integer::intValue).toArray();
		} else if (order.length != n) {
			throw new RuntimeException("length of order has to equal n");
		}

		// Boolean sparsity mask under the column reordering: nz[i][j] = (A[i, order[j]] != 0).
		boolean[][] nz = new boolean[m][n];
		for (int j = 0; j < n; ++j) {
			int origCol = order[j];
			for (int i = 0; i < m; ++i) {
				nz[i][j] = A.getAsDouble(i, origCol) != 0.0;
			}
		}

		int[] groups = greedyColumnGrouping(m, n, nz);

		int[] orderedGroups = new int[n];
		for (int i = 0; i < n; ++i) orderedGroups[order[i]] = groups[i];
		return orderedGroups;
	}

	/**
	 * Curtis-Powell-Reid greedy column grouping on the boolean sparsity mask.
	 * Two columns are in the same group iff their non-zero rows are disjoint.
	 */
	private static int[] greedyColumnGrouping(int m, int n, boolean[][] nz) {
		int[] groups = new int[n];
		Arrays.fill(groups, -1);

		boolean[] union = new boolean[m];
		int currentGroup = 0;

		for (int i = 0; i < n; ++i) {
			if (groups[i] >= 0) continue;
			groups[i] = currentGroup;
			boolean allGrouped = true;

			// Reset union to column i's mask.
			for (int k = 0; k < m; ++k) union[k] = nz[k][i];

			for (int j = i + 1; j < n; ++j) {
				if (groups[j] >= 0) continue;
				allGrouped = false;

				boolean intersect = false;
				for (int k = 0; k < m; ++k) {
					if (union[k] && nz[k][j]) { intersect = true; break; }
				}
				if (!intersect) {
					for (int k = 0; k < m; ++k) {
						if (nz[k][j]) union[k] = true;
					}
					groups[j] = currentGroup;
				}
			}

			if (allGrouped) break;
			currentGroup++;
		}
		return groups;
	}

	/**
	 * Adjust finite-difference step sizes so that the perturbed point stays within
	 * {@code [lb, ub]} per dimension, flipping direction or switching forward / backward
	 * as needed. Pure per-element logic on length-{@code n} vectors.
	 *
	 * @param x0       point at which to estimate the derivative ({@code n x 1})
	 * @param h        desired absolute finite-difference steps ({@code n x 1})
	 * @param numSteps number of {@code h} steps taken per direction
	 * @param scheme   one-sided or two-sided differencing
	 * @param lb       per-variable lower bounds ({@code n x 1})
	 * @param ub       per-variable upper bounds ({@code n x 1})
	 * @return adjusted step sizes plus per-dimension one-sided fall-back flags
	 */
	public static AdjustedDifferencingScheme adjustSchemeToBounds(Matrix x0, Matrix h, int numSteps,
			FiniteDifferenceMethod scheme, Matrix lb, Matrix ub) {
		int n = (int) h.getRowCount();

		double[] x0Arr = x0.toColumnArray();
		double[] hArr = h.toColumnArray();
		double[] lbArr = lb.toColumnArray();
		double[] ubArr = ub.toColumnArray();

		boolean[] useOneSided = new boolean[n];
		switch (scheme) {
		case ONE_SIDED:
			Arrays.fill(useOneSided, true);
			break;
		case TWO_SIDED:
			for (int i = 0; i < n; ++i) hArr[i] = Math.abs(hArr[i]);
			// useOneSided already false-initialised
			break;
		default:
			throw new RuntimeException("schema must be either ONE_SIDED or TWO_SIDED");
		}

		boolean hasBounds = false;
		for (int i = 0; i < n; ++i) {
			if (lbArr[i] != Double.NEGATIVE_INFINITY || ubArr[i] != Double.POSITIVE_INFINITY) {
				hasBounds = true;
				break;
			}
		}
		if (!hasBounds) {
			return new AdjustedDifferencingScheme(DenseMatrix.column(hArr), useOneSided);
		}

		double[] hAdjusted = hArr.clone();

		if (scheme == FiniteDifferenceMethod.ONE_SIDED) {
			for (int i = 0; i < n; ++i) {
				double lowerDist = x0Arr[i] - lbArr[i];
				double upperDist = ubArr[i] - x0Arr[i];
				double maxDist = Math.max(lowerDist, upperDist);
				double hTotal = hArr[i] * numSteps;
				double xi = x0Arr[i] + hTotal;
				boolean violated = xi < lbArr[i] || xi > ubArr[i];
				boolean fitting = Math.abs(hTotal) <= maxDist;
				if (violated && fitting) {
					hAdjusted[i] = -hAdjusted[i];
				}
				if (!fitting) {
					if (upperDist >= lowerDist) {
						hAdjusted[i] = upperDist / numSteps;
					} else {
						hAdjusted[i] = -lowerDist / numSteps;
					}
				}
			}
		} else { // TWO_SIDED
			for (int i = 0; i < n; ++i) {
				double lowerDist = x0Arr[i] - lbArr[i];
				double upperDist = ubArr[i] - x0Arr[i];
				double hTotal = hArr[i] * numSteps;
				boolean central = lowerDist >= hTotal && upperDist >= hTotal;
				if (!central) {
					if (upperDist >= lowerDist) {
						hAdjusted[i] = Math.min(hArr[i], 0.5 * upperDist / numSteps);
					} else {
						hAdjusted[i] = -Math.min(hArr[i], 0.5 * lowerDist / numSteps);
					}
					useOneSided[i] = true;
				}
				double minDist = Math.min(upperDist, lowerDist) / numSteps;
				if (!central && Math.abs(hAdjusted[i]) <= minDist) {
					hAdjusted[i] = minDist;
					useOneSided[i] = false;
				}
			}
		}

		return new AdjustedDifferencingScheme(DenseMatrix.column(hAdjusted), useOneSided);
	}

	/**
	 * Calculate the relative EPS step for a given data type and FD method.
	 * Progressively smaller steps are used for larger floating-point types.
	 * If {@code x0} or {@code f0} is {@code float}, the smaller (less precise)
	 * machine epsilon is chosen.
	 *
	 * @param x0Type element type of {@code x0} ({@code float} or {@code double})
	 * @param f0Type element type of {@code f0} ({@code float} or {@code double})
	 * @param method FD scheme: TWO_POINT, THREE_POINT, or COMPLEX_STEP
	 * @return relative step size for this combination
	 */
	public static double epsForMethod(Class<?> x0Type, Class<?> f0Type, FiniteDifferenceMethod method) {

		double EPS = Math.ulp(1.0);

		final boolean x0IsFp;
		final int x0ItemSize;
		if (x0Type.equals(float.class) || x0Type.equals(Float.class)) {
			EPS = Math.ulp((float) 1.0);
			x0ItemSize = Float.BYTES;
			x0IsFp = true;
		} else if (x0Type.equals(double.class) || x0Type.equals(Double.class)) {
			EPS = Math.ulp((double) 1.0);
			x0ItemSize = Double.BYTES;
			x0IsFp = true;
		} else {
			x0ItemSize = Integer.MAX_VALUE;
			x0IsFp = false;
		}

		final boolean f0IsFp;
		final int f0ItemSize;
		final double f0Eps;
		if (f0Type.equals(float.class) || f0Type.equals(Float.class)) {
			f0ItemSize = Float.BYTES;
			f0IsFp = true;
			f0Eps = Math.ulp((float) 1.0);
		} else if (f0Type.equals(double.class) || f0Type.equals(Double.class)) {
			f0ItemSize = Double.BYTES;
			f0IsFp = true;
			f0Eps = Math.ulp((double) 1.0);
		} else {
			f0ItemSize = Integer.MAX_VALUE;
			f0IsFp = false;
			f0Eps = Double.NaN;
		}

		if (x0IsFp && f0IsFp && f0ItemSize < x0ItemSize) {
			EPS = f0Eps;
		}

		switch (method) {
		case TWO_POINT: // fall-through
		case COMPLEX_STEP:
			return Math.sqrt(EPS);
		case THREE_POINT:
			return Math.pow(EPS, 1.0 / 3.0);
		default:
			throw new RuntimeException("only implemented for TWO_POINT, COMPLEX_STEP and THREE_POINT");
		}
	}

	/**
	 * Compute the absolute finite-difference step from a relative step.
	 *
	 * @param relStep per-variable relative step ({@code n x 1}); pass
	 *                {@code null} to derive a default from {@code epsForMethod}
	 * @param x0      point of evaluation ({@code n x 1})
	 * @param f0      function value at {@code x0} ({@code m x 1}); used only
	 *                for value-type dispatch in this port (always treated as
	 *                {@code double})
	 * @param method  FD scheme: TWO_POINT, THREE_POINT, or COMPLEX_STEP
	 * @return absolute step ({@code n x 1})
	 */
	public static Matrix computeAbsoluteStep(Matrix relStep, Matrix x0, Matrix f0, FiniteDifferenceMethod method) {
		double[] x0Arr = x0.toColumnArray();
		int n = x0Arr.length;

		// sign_x0[i] = +1 if x0[i] >= 0 else -1 (so 0 maps to +1, matching scipy).
		double[] x0Sign = new double[n];
		for (int i = 0; i < n; ++i) x0Sign[i] = x0Arr[i] >= 0.0 ? 1.0 : -1.0;

		// This port stores all matrices as double; no need to dispatch on value type.
		double rStep = epsForMethod(double.class, double.class, method);

		double[] absStepArr = new double[n];
		if (relStep == null) {
			for (int i = 0; i < n; ++i) {
				absStepArr[i] = rStep * x0Sign[i] * Math.max(1.0, Math.abs(x0Arr[i]));
			}
		} else {
			double[] relStepArr = relStep.toColumnArray();
			// User has requested specific relative steps. Don't multiply by max(1, abs(x0))
			// because if x0 < 1 then their requested step is not used.
			for (int i = 0; i < n; ++i) {
				absStepArr[i] = relStepArr[i] * x0Sign[i] * Math.abs(x0Arr[i]);
				// We don't want an abs_step of 0, which can happen if rel_step is 0 or x0 is 0.
				// In that case fall back to the auto-step.
				double dx = (x0Arr[i] + absStepArr[i]) - x0Arr[i];
				if (dx == 0.0) {
					absStepArr[i] = rStep * x0Sign[i] * Math.max(1.0, Math.abs(x0Arr[i]));
				}
			}
		}

		return DenseMatrix.column(absStepArr);
	}

	/**
	 * Finite-difference gradient of a scalar function of a scalar argument.
	 *
	 * @param f       scalar function {@code R -> R}
	 * @param x0      evaluation point
	 * @param options FD configuration
	 * @return scalar derivative {@code df/dx} at {@code x0}
	 */
	public static double approxDerivative(DoubleUnaryOperator f, double x0, FiniteDifferenceOptions options) {
		UnaryOperator<Matrix> fVec = (Matrix t) -> DenseMatrix.column(f.applyAsDouble(t.doubleValue()));
		Matrix x0Vec = DenseMatrix.column(x0);
		return approxDerivative(fVec, x0Vec, null, options).doubleValue();
	}

	/**
	 * Finite-difference gradient of a {@link ToDoubleFunction} of a scalar argument.
	 *
	 * @param f       scalar function evaluated on a {@code 1 x 1} {@link Matrix}
	 * @param x0      evaluation point
	 * @param options FD configuration
	 * @return scalar derivative {@code df/dx} at {@code x0}
	 */
	public static double approxDerivative(ToDoubleFunction<Matrix> f, double x0, FiniteDifferenceOptions options) {
		UnaryOperator<Matrix> fVec = (Matrix t) -> DenseMatrix.column(f.applyAsDouble(t));
		Matrix x0Vec = DenseMatrix.column(x0);
		return approxDerivative(fVec, x0Vec, null, options).doubleValue();
	}

	/**
	 * Finite-difference gradient of a scalar function of a scalar argument with a precomputed {@code f0}.
	 *
	 * @param f       scalar function evaluated on a {@code 1 x 1} {@link Matrix}
	 * @param x0      evaluation point
	 * @param f0      precomputed {@code f(x0)} (saves one evaluation)
	 * @param options FD configuration
	 * @return scalar derivative {@code df/dx} at {@code x0}
	 */
	public static double approxDerivative(ToDoubleFunction<Matrix> f, double x0, double f0, FiniteDifferenceOptions options) {
		Function<Matrix, Matrix> fVec = (Matrix t) -> DenseMatrix.column(f.applyAsDouble(t));
		Matrix x0Vec = DenseMatrix.column(x0);
		Matrix f0Vec = DenseMatrix.column(f0);
		return approxDerivative(fVec, x0Vec, f0Vec, options).doubleValue();
	}

	/**
	 * Finite-difference gradient of a real-valued function of a vector argument.
	 *
	 * @param f       function {@code R^n -> R}
	 * @param x0      evaluation point ({@code n x 1})
	 * @param f0      precomputed {@code f(x0)}
	 * @param options FD configuration
	 * @return gradient {@code gradf(x0)} ({@code 1 x n} or {@code n x 1} depending on
	 *         the underlying call shape)
	 */
	public static Matrix approxDerivative(ToDoubleFunction<Matrix> f, Matrix x0, double f0, FiniteDifferenceOptions options) {
		Function<Matrix, Matrix> fVec = (Matrix t) -> DenseMatrix.column(f.applyAsDouble(t));
		Matrix f0Vec = DenseMatrix.column(f0);
		return approxDerivative(fVec, x0, f0Vec, options);
	}

	/**
	 * Finite-difference Jacobian of a vector-valued function.
	 *
	 * @param f       function {@code R^n -> R^m}
	 * @param x0      evaluation point ({@code n x 1})
	 * @param f0      precomputed {@code f(x0)} ({@code m x 1}); pass {@code null}
	 *                to evaluate inside
	 * @param options FD configuration
	 * @return Jacobian {@code J} ({@code m x n})
	 */
	public static Matrix approxDerivative(Function<Matrix, Matrix> f, Matrix x0, Matrix f0, FiniteDifferenceOptions options) {
		BiFunction<Matrix, Object, Matrix> fArg = (Matrix t, Object _args) -> f.apply(t);
		return approxDerivative(fArg, x0, f0, options, null);
	}

	/**
	 * Finite-difference Jacobian of a vector-valued function with extra args.
	 * See the class doc / scipy {@code _numdiff.approx_derivative} for the full
	 * specification.
	 *
	 * @param fun     function {@code (x, args) -> f(x, args)}
	 * @param x0      evaluation point ({@code n x 1})
	 * @param f0      precomputed {@code fun(x0, args)} ({@code m x 1}); pass
	 *                {@code null} to evaluate inside
	 * @param options FD configuration
	 * @param args    extra arguments forwarded to {@code fun}
	 * @return Jacobian {@code J} ({@code m x n}); a {@link LinearOperator} when
	 *         {@code options.asLinearOperator()} is {@code true}
	 */
	public static Matrix approxDerivative(BiFunction<Matrix, Object, Matrix> fun, Matrix x0, Matrix f0,
			FiniteDifferenceOptions options, Object args) {

		switch (options.method()) {
		case TWO_POINT: case THREE_POINT: case COMPLEX_STEP:
			break;
		default:
			throw new RuntimeException("Can only use TWO_POINT, THREE_POINT or COMPLEX_STEP");
		}

		if (x0.getSize().length != 2 || x0.getColumnCount() != 1) {
			throw new RuntimeException("x0 must be of shape (n,1)");
		}

		Matrix lb = options.bounds().lb();
		Matrix ub = options.bounds().ub();
		if (lb.getRowCount() != x0.getRowCount() || lb.getColumnCount() != x0.getColumnCount()
				|| ub.getRowCount() != x0.getRowCount() || ub.getColumnCount() != x0.getColumnCount()) {
			throw new RuntimeException("Inconsistent shapes between bounds and `x0`.");
		}

		int n = (int) x0.getRowCount();

		if (options.asLinearOperator()) {
			for (int i = 0; i < n; ++i) {
				double l = lb.getAsDouble(i, 0);
				double u = ub.getAsDouble(i, 0);
				if (l != Double.NEGATIVE_INFINITY || u != Double.POSITIVE_INFINITY) {
					throw new RuntimeException("Bounds not supported when `as_linear_operator` is True.");
				}
			}
		}

		// `x0` violates bound constraints?
		for (int i = 0; i < n; ++i) {
			double xi = x0.getAsDouble(i, 0);
			if (xi < lb.getAsDouble(i, 0) || xi > ub.getAsDouble(i, 0)) {
				throw new RuntimeException("`x0` violates bound constraints.");
			}
		}

		UnaryOperator<Matrix> funWrapped = x -> fun.apply(x, args);

		if (f0 == null) {
			f0 = funWrapped.apply(x0);
		}

		if (options.asLinearOperator()) {
			final Matrix relStep;
			if (options.relStep() == null) {
				relStep = Matrix.Factory.ones(x0.getSize()).times(epsForMethod(double.class, double.class, options.method()));
			} else {
				relStep = options.relStep();
			}
			return (Matrix) linearOperatorDifference(funWrapped, x0, f0, relStep, options.method());
		}

		// Default: relative step.
		final Matrix absStep;
		if (options.absStep() == null) {
			absStep = computeAbsoluteStep(options.relStep(), x0, f0, options.method());
		} else {
			// User specified an absolute step. Cannot have a zero step;
			// fall back to the auto step if (x0+abs)-x0 underflows to zero.
			absStep = options.absStep();
			double rStep = epsForMethod(double.class, double.class, options.method());
			for (int i = 0; i < n; ++i) {
				double xi = x0.getAsDouble(i, 0);
				double si = absStep.getAsDouble(i, 0);
				double dxVal = (xi + si) - xi;
				if (dxVal == 0.0) {
					double sign = xi >= 0.0 ? 1.0 : -1.0;
					absStep.setAsDouble(rStep * sign * Math.max(1.0, Math.abs(xi)), i, 0);
				}
			}
		}

		final AdjustedDifferencingScheme ads;
		switch (options.method()) {
		case TWO_POINT:
			ads = adjustSchemeToBounds(x0, absStep, 1, FiniteDifferenceMethod.ONE_SIDED, lb, ub);
			break;
		case THREE_POINT:
			ads = adjustSchemeToBounds(x0, absStep, 1, FiniteDifferenceMethod.TWO_SIDED, lb, ub);
			break;
		case COMPLEX_STEP:
			throw new RuntimeException("not implemented yet");
		default:
			throw new RuntimeException("only TWO_POINT, THREE_POINT and COMPLEX_STEP are allowed");
		}

		if (options.sparsity() == null) {
			return denseDifference(funWrapped, x0, f0, ads.hAdjusted(), ads.useOneSided(), options.method());
		}
		return sparseDifference(funWrapped, x0, f0, ads.hAdjusted(), ads.useOneSided(), options.sparsity(), options.method());
	}

	/**
	 * Build a {@link LinearOperator} that applies the FD Jacobian
	 * {@code J(x0)*p} on demand, without materialising the full Jacobian.
	 * Used by {@code as_linear_operator=True} consumers.
	 *
	 * @param fun     vector-valued function being differentiated
	 * @param x0      evaluation point
	 * @param f0      cached {@code fun(x0)}
	 * @param relStep relative step size
	 * @param method  FD scheme (TWO_POINT, THREE_POINT, COMPLEX_STEP)
	 * @return matrix-free FD-Jacobian linear operator
	 */
	public static LinearOperator linearOperatorDifference(Function<Matrix, Matrix> fun,
			Matrix x0, Matrix f0, Matrix relStep, FiniteDifferenceMethod method) {

		switch (method) {
		case TWO_POINT:
			return (Matrix p) -> {
				if (p.normInf() == 0.0) {
					return Matrix.Factory.zeros(f0.getSize());
				}
				Matrix dx = relStep.divide(p.norm2());
				Matrix x = x0.plus(dx.times(relStep));
				Matrix df = fun.apply(x).minus(f0);
				return df.divide(dx);
			};
		case THREE_POINT:
			return (Matrix p) -> {
				if (p.normInf() == 0.0) {
					return Matrix.Factory.zeros(f0.getSize());
				}
				Matrix dx = relStep.times(2.0).divide(p.norm2());
				Matrix x1 = x0.minus(dx.divide(2.0).times(p));
				Matrix x2 = x0.plus(dx.divide(2.0).times(p));
				Matrix f1 = fun.apply(x1);
				Matrix f2 = fun.apply(x2);
				Matrix df = f2.minus(f1);
				return df.divide(dx);
			};
		case COMPLEX_STEP:
			return (Matrix p) -> {
				if (p.normInf() == 0.0) {
					return Matrix.Factory.zeros(f0.getSize());
				}
				throw new RuntimeException("not implemented yet");
			};
		default:
			throw new RuntimeException("only TWO_POINT, THREE_POINT and COMPLEX_STEP are allowed");
		}
	}

	private static Matrix denseDifference(Function<Matrix, Matrix> fun, Matrix x0, Matrix f0,
			Matrix absStep, boolean[] useOneSided, FiniteDifferenceMethod method) {

		int m = (int) f0.getRowCount();
		int n = (int) x0.getRowCount();

		Matrix Jt = Matrix.Factory.zeros(n, m);
		double[] absStepArr = absStep.toColumnArray();

		// Per-column directional step vector built lazily inside the loop.
		for (int i = 0; i < n; ++i) {
			double hi = absStepArr[i];

			// hI = e_i * hi (only the i-th component is non-zero).
			Matrix hI = Matrix.Factory.zeros(n, 1);
			hI.setAsDouble(hi, i, 0);

			final double dx;
			final Matrix df;
			switch (method) {
			case TWO_POINT: {
				Matrix x = x0.plus(hI);
				dx = x.getAsDouble(i, 0) - x0.getAsDouble(i, 0);
				df = fun.apply(x).minus(f0);
				break;
			}
			case THREE_POINT:
				if (useOneSided[i]) {
					Matrix x1 = x0.plus(hI);
					Matrix x2 = x0.plus(hI.times(2.0));
					dx = x2.getAsDouble(i, 0) - x0.getAsDouble(i, 0);
					Matrix f1 = fun.apply(x1);
					Matrix f2 = fun.apply(x2);
					df = f0.times(-3.0).plus(f1.times(4.0)).minus(f2);
				} else {
					Matrix x1 = x0.minus(hI);
					Matrix x2 = x0.plus(hI);
					dx = x2.getAsDouble(i, 0) - x1.getAsDouble(i, 0);
					Matrix f1 = fun.apply(x1);
					Matrix f2 = fun.apply(x2);
					df = f2.minus(f1);
				}
				break;
			case COMPLEX_STEP:
				throw new RuntimeException("not implemented yet");
			default:
				throw new RuntimeException("only TWO_POINT, THREE_POINT and COMPLEX_STEP are allowed");
			}

			for (int p = 0; p < m; ++p) {
				Jt.setAsDouble(df.getAsDouble(p, 0) / dx, i, p);
			}
		}

		return Jt.transpose();
	}

	private static Matrix sparseDifference(Function<Matrix, Matrix> fun, Matrix x0, Matrix f0,
			Matrix absStep, boolean[] useOneSided, Sparsity sparsity, FiniteDifferenceMethod method) {

		int m = (int) f0.getRowCount();
		int n = (int) x0.getRowCount();

		Matrix J = SparseMatrix.Factory.zeros(m, n);

		int[] groups = sparsity.sparsityGroups();
		int nGroups = Arrays.stream(groups).max().getAsInt() + 1;
		Matrix structure = sparsity.A();

		double[] absStepArr = absStep.toColumnArray();

		for (int group = 0; group < nGroups; ++group) {
			// Variables in the current group get a non-zero step; others stay at 0.
			boolean[] inGroup = new boolean[n];
			double[] hArr = new double[n];
			for (int i = 0; i < n; ++i) {
				if (groups[i] == group) {
					inGroup[i] = true;
					hArr[i] = absStepArr[i];
				}
			}
			Matrix h = DenseMatrix.column(hArr);

			if (method == FiniteDifferenceMethod.TWO_POINT) {
				Matrix x = x0.plus(h);
				Matrix dx = x.minus(x0);
				Matrix df = fun.apply(x).minus(f0);
				// J[i, j] = df[i] / dx[j] for every (i, j) in the sparsity pattern with j in this group.
				for (int j = 0; j < n; ++j) {
					if (!inGroup[j]) continue;
					double dxj = dx.getAsDouble(j, 0);
					for (int i = 0; i < m; ++i) {
						if (structure.getAsDouble(i, j) != 0.0) {
							J.setAsDouble(df.getAsDouble(i, 0) / dxj, i, j);
						}
					}
				}
			} else if (method == FiniteDifferenceMethod.THREE_POINT) {
				// Mixed one-sided / central per dimension.
				Matrix x1 = Matrix.Factory.zeros(n, 1);
				Matrix x2 = Matrix.Factory.zeros(n, 1);
				Matrix dx = Matrix.Factory.zeros(n, 1);
				for (int i = 0; i < n; ++i) {
					if (!inGroup[i]) continue;
					double x0i = x0.getAsDouble(i, 0);
					double hi = h.getAsDouble(i, 0);
					double x1Val, x2Val;
					if (useOneSided[i]) {
						x1Val = x0i + hi;
						x2Val = x0i + 2.0 * hi;
					} else {
						x1Val = x0i - hi;
						x2Val = x0i + hi;
					}
					x1.setAsDouble(x1Val, i, 0);
					x2.setAsDouble(x2Val, i, 0);
					dx.setAsDouble(x2Val - x1Val, i, 0);
				}

				Matrix f1 = fun.apply(x1);
				Matrix f2 = fun.apply(x2);

				double[] dfArr = new double[m];
				for (int j = 0; j < m; ++j) {
					if (useOneSided[j]) {
						dfArr[j] = -3.0 * f0.getAsDouble(j, 0) + 4.0 * f1.getAsDouble(j, 0) - f2.getAsDouble(j, 0);
					} else {
						dfArr[j] = f2.getAsDouble(j, 0) - f1.getAsDouble(j, 0);
					}
				}

				for (int j = 0; j < n; ++j) {
					if (!inGroup[j]) continue;
					double dxj = dx.getAsDouble(j, 0);
					for (int i = 0; i < m; ++i) {
						if (structure.getAsDouble(i, j) != 0.0) {
							J.setAsDouble(dfArr[i] / dxj, i, j);
						}
					}
				}
			} else if (method == FiniteDifferenceMethod.COMPLEX_STEP) {
				throw new RuntimeException("not implemented yet");
			} else {
				throw new RuntimeException("only TWO_POINT, THREE_POINT and COMPLEX_STEP are allowed");
			}
		}
		return J;
	}
}
