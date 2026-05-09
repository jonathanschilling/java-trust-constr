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
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;

public class NumDiff {

	/**
	 * Group columns of a 2-D matrix for sparse finite differencing [1].
	 *
	 * Two columns are in the same group if in each row at least one of them has
	 * zero. A greedy sequential algorithm is used to construct groups.
	 *
	 * @see [1] A. Curtis, M. J. D. Powell, and J. Reid,
	 *      "On the estimation of sparse Jacobian matrices", Journal of the
	 *      Institute of Mathematics and its Applications, 13 (1974), pp. 117-120.
	 *
	 * @param A [m][n] Matrix of which to group columns.
	 * @return [n] Contains values from 0 to n_groups-1, where n_groups is the
	 *         number of found groups. Each value ``groups[i]`` is an index of a
	 *         group to which ith column assigned. The procedure was helpful only if
	 *         n_groups is significantly less than n.
	 */
	public static int[] groupColumns(Matrix A) {
		int[] defaultOrder = { 0 };
		return groupColumns(A, defaultOrder);
	}

	/**
	 * Group columns of a 2D matrix for sparse finite differencing [1].
	 *
	 * Two columns are in the same group if in each row at least one of them has zero.
	 * A greedy sequential algorithm is used to construct groups.
	 *
	 * @see [1] A. Curtis, M. J. D. Powell, and J. Reid,
	 *          "On the estimation of sparse Jacobian matrices",
	 *          Journal of the Institute of Mathematics and its Applications, 13 (1974), pp. 117-120.
	 *
	 * @param A     [m][n] Matrix of which to group columns.
	 * @param order [n] Permutation array which defines the order of columns enumeration.
	 *                  If int or None, a random permutation is used with `order` used as a random seed.
	 *                  Default is 0, that is use a random permutation but guarantee repeatability.
	 * @return [n] Contains values from 0 to n_groups-1, where n_groups is the number of found groups.
	 *             Each value ``groups[i]`` is an index of a group to which the i-th is column assigned.
	 *             The procedure was helpful only if n_groups is significantly less than n.
	 */
	public static int[] groupColumns(Matrix A, int[] order) {

		// TODO: is this a sparse int matrix already?
		Matrix newA = SparseMatrix.Factory.zeros(A.getSize());
		for (long[] pos: A.allCoordinates()) {
			double aVal = A.getAsDouble(pos);
			if (aVal != 0.0) {
				newA.setAsInt(1, pos);
			}
		}
		A = newA;

		if (A.getSize().length != 2) {
			throw new RuntimeException("`A` must be 2-dimensional.");
		}

		long m = A.getRowCount();
		long n = A.getColumnCount();

		// get random, but reproducible order if no order is given
		// or check given order for compatibility with A
		if (order == null || order.length <= 1) {
			// reproducible RNG
			Random rnd;
			if (order == null || order.length == 0) {
				rnd = new Random(0);
			} else {
				rnd = new Random(order[0]);
			}

			// obtain permutation of [0, 1, ..., (n-1)]
			List<Integer> indices = new ArrayList<>((int) n);
			for (int i = 0; i < n; ++i) {
				indices.add(i);
			}
			Collections.shuffle(indices, rnd);
			order = indices.stream().mapToInt(i -> i).toArray();
		} else {
			if (order.length != n) {
				throw new RuntimeException("length of order has to equal n");
			}
		}

		// apply column ordering
		// TODO: this surely can be done more elegantly...
		Matrix orderedA = SparseMatrix.Factory.zeros(A.getSize());
		for (long[] pos: A.allCoordinates()) {
			// Take elements from column given in order[col] ...
			double aVal = A.getAsDouble(pos[0], order[(int) pos[1]]);
			if (aVal != 0.0) { // retain sparsity
				// ... and put them into column col.
				orderedA.setAsDouble(aVal, pos);
			}
		}
		A = orderedA;

		final int[] groups;
		if (A.isSparse()) {
			groups = groupSparse((int) m, (int) n, (SparseMatrix) A);
		} else {
			groups = groupDense((int) m, (int) n, A);
		}

		final int[] orderedGroups = new int[groups.length];
		for (int i = 0; i<n; ++i) {
			orderedGroups[order[i]] = groups[i];
		}

		return orderedGroups;
	}

	private static int[] groupDense(int m, int n, Matrix A) {

		int[] groups = new int[n];
		Arrays.fill(groups, -1);

		int currentGroup = 0;

		int[] union = new int[m];

		// Loop through all the columns.
		for (int i=0; i<n; ++i) {
			if (groups[i] >= 0) {
				// A group was already assigned.
				continue;
			}

			groups[i] = currentGroup;
			boolean allGrouped = true;

			// Here we store the union of grouped columns.
			Matrix aCol = A.selectColumns(i);
			for (long[] pos: aCol.allCoordinates()) {
				union[(int) pos[1]] = aCol.getAsInt(pos);
			}

			for (int j = 0; j < n; ++j) {
				if (groups[j] < 0) {
					allGrouped = false;
				} else {
					continue;
				}

				// Determine if j-th column intersects with the union.
				boolean intersect = false;
				for (int k=0; k<m; ++k) {
					if (union[k] > 0 && A.getAsInt(k, j) > 0) {
						intersect = true;
						break;
					}
				}

				// If not, add it to the union and assign the group to it.
				if (!intersect) {
					Matrix aOtherCol = A.selectColumns(j);
					for (long[] pos: aOtherCol.allCoordinates()) {
						union[(int) pos[1]] += aOtherCol.getAsInt(pos);
					}
					groups[j] = currentGroup;
				}
			}

			if (allGrouped) {
				break;
			}

			currentGroup++;
		}

		return groups;
	}

	/**
	 * Find groups of independent columns of a sparse matrix.
	 *
	 * @see https://stackoverflow.com/a/52299730
	 *
	 * @param m
	 * @param n
	 * @param A
	 * @return
	 */
	private static int[] groupSparse(int m, int n, SparseMatrix A) {
		int[] groups = new int[n];
		Arrays.fill(groups, -1);

		int currentGroup = 0;

		int[] union = new int[m];

		// Loop through all the columns.
		for (int i=0; i<n; ++i) {
			if (groups[i] >= 0) {
				// A group was already assigned.
				continue;
			}

			groups[i] = currentGroup;
			boolean allGrouped = true;

			// Here we store the union of grouped columns.
			Arrays.fill(union, 0);
			Matrix ithCol = A.subMatrix(0, i, A.getRowCount()-1, i);
			for (long[] pos: ithCol.availableCoordinates()) {
				if (ithCol.getAsDouble(pos) != 0.0) {
					union[(int) pos[0]] = 1;
				}
			}

			for (int j = 0; j < n; ++j) {
				if (groups[j] < 0) {
					allGrouped = false;
				} else {
					continue;
				}

				// Determine if j-th column intersects with the union.
				boolean intersect = false;
				Matrix jthCol = A.subMatrix(0, j, A.getRowCount()-1, j);
				for (long[] pos: jthCol.availableCoordinates()) {
					if (jthCol.getAsDouble(pos) != 0.0) {
						if (union[(int) pos[0]] == 1) {
							intersect = true;
							break;
						}
					}
				}

				// If not, add it to the union and assign the group to it.
				if (!intersect) {
					for (long[] pos: jthCol.availableCoordinates()) {
						if (jthCol.getAsDouble(pos) != 0.0) {
							union[(int) pos[0]] = 1;
						}
					}
					groups[j] = currentGroup;
				}
			}

			if (allGrouped) {
				break;
			}

			currentGroup++;
		}

		return groups;
	}

	/**
	 * @param x0       [n] Point at which we wish to estimate derivative.
	 * @param h        [n] Desired absolute finite difference steps.
	 * @param numSteps Number of `h` steps in one direction required to implement
	 *                 finite difference scheme. For example, 2 means that we need
	 *                 to evaluate f(x0 + 2 * h) or f(x0 - 2 * h)
	 * @param scheme   Whether steps in one or both directions are required. In
	 *                 other words '1-sided' applies to forward and backward
	 *                 schemes, '2-sided' applies to center schemes.
	 * @param lb       [n] Lower bounds on independent variables.
	 * @param ub       [n] Upper bounds on independent variables.
	 * @return hAdjusted Adjusted absolute step sizes.
	 *                   Step size decreases only if a sign flip
	 *                   or switching to one-sided scheme doesn't allow to take a full step.
	 *         useOneSided Whether to switch to one-sided scheme.
	 *                     Informative only for ``scheme='2-sided'``.
	 */
	public static AdjustedDifferencingScheme adjustSchemeToBounds(Matrix x0, Matrix h, int numSteps,
			FiniteDifferenceMethod scheme, Matrix lb, Matrix ub) {

		int n = (int) h.getRowCount();

		final boolean[] useOneSided = new boolean[n];
		switch (scheme) {
		case ONE_SIDED:
			Arrays.fill(useOneSided, true);
			break;
		case TWO_SIDED:
			h = h.absInPlace();
			Arrays.fill(useOneSided, false); // could be omitted...
			break;
		default:
			throw new RuntimeException("schema must be either ONE_SIDED or TWO_SIDED");
		}

		boolean hasBounds = false;
		for (int i=0; i<n; ++i) {
			if (lb.getAsDouble(i, 0) != Double.NEGATIVE_INFINITY || ub.getAsDouble(i, 0) != Double.POSITIVE_INFINITY) {
				hasBounds = true;
				break;
			}
		}
		if (!hasBounds) {
			return new AdjustedDifferencingScheme(h, useOneSided);
		}

		Matrix hTotal = h.times(numSteps);
		Matrix hAdjusted = Matrix.Factory.copyFromMatrix(h);

		Matrix lowerDist = x0.minus(lb);
		Matrix upperDist = ub.minus(x0);

		Matrix maxDist = Matrix.Factory.copyFromMatrix(lowerDist);
		for (long[] pos: maxDist.allCoordinates()) {
			double ud = upperDist.getAsDouble(pos);
			if (ud > maxDist.getAsDouble(pos)) {
				maxDist.setAsDouble(ud, pos);
			}
		}

		if (scheme == FiniteDifferenceMethod.ONE_SIDED) {
			Matrix x = x0.plus(hTotal);
			Matrix violated = x.lt(lb).or(x.gt(ub));
			Matrix fitting = hTotal.abs().le(maxDist);
			for (long[] pos: hAdjusted.allCoordinates()) {
				if (violated.getAsBoolean(pos) && fitting.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(-1.0 * hAdjusted.getAsDouble(pos), pos);
				}
			}

			Matrix forward = upperDist.ge(lowerDist).and(fitting.not());
			for (long[] pos: forward.availableCoordinates()) {
				if (forward.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(upperDist.getAsDouble(pos) / numSteps, pos);
				}
			}

			Matrix backward = upperDist.lt(lowerDist).and(fitting.not());
			for (long[] pos: backward.availableCoordinates()) {
				if (backward.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(-lowerDist.getAsDouble(pos) / numSteps, pos);
				}
			}

		} else if (scheme == FiniteDifferenceMethod.TWO_SIDED) {
			Matrix central = lowerDist.ge(hTotal).and(upperDist.ge(hTotal));

			Matrix forward = upperDist.ge(lowerDist).and(central.not());
			for (long[] pos: forward.availableCoordinates()) {
				if (forward.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(Math.min(h.getAsDouble(pos), 0.5 * upperDist.getAsDouble(pos) / numSteps), pos);
					useOneSided[(int) pos[0]] = true;
				}
			}

			Matrix backward = upperDist.lt(lowerDist).and(central.not());
			for (long[] pos: backward.availableCoordinates()) {
				if (backward.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(-1.0 * Math.min(h.getAsDouble(pos), 0.5 * lowerDist.getAsDouble(pos) / numSteps), pos);
					useOneSided[(int) pos[0]] = true;
				}
			}

			Matrix minDist = Matrix.Factory.zeros(upperDist.getSize());
			for (long[] pos: upperDist.allCoordinates()) {
				minDist.setAsDouble(Math.min(upperDist.getAsDouble(pos), lowerDist.getAsDouble(pos)) / numSteps, pos);
			}
			Matrix adjustedCentral = central.not().and(hAdjusted.abs().le(minDist));
			for (long[] pos: adjustedCentral.availableCoordinates()) {
				if (adjustedCentral.getAsBoolean(pos)) {
					hAdjusted.setAsDouble(minDist.getAsDouble(pos), pos);
					useOneSided[(int) pos[0]] = false;
				}
			}
		} else {
			throw new RuntimeException("schema must be either ONE_SIDED or TWO_SIDED");
		}

		return new AdjustedDifferencingScheme(hAdjusted, useOneSided);
	}

	/**
	 * Calculates relative EPS step to use for a given data type and numdiff step method.
     *
     * Progressively smaller steps are used for larger floating point types.
     *
     * The default relative step will be double.
     * However, if x0 or f0 are smaller floating point types (float),
     * then the smallest floating point type is chosen.
     *
	 * @param x0Type type of parameter vector
	 * @param f0Type type of function evaluation
	 * @param method {'2-point', '3-point', 'cs'}
	 * @return relative step size to use
	 */
	public static double epsForMethod(Class<?> x0Type, Class<?> f0Type, FiniteDifferenceMethod method) {

		// the default EPS value
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
			// choose the smallest itemsize between x0 and f0
			EPS = f0Eps;
		}

		switch (method) {
		case TWO_POINT: // fall-through
		case COMPLEX_STEP:
			return Math.sqrt(EPS);
		case THREE_POINT:
			return Math.pow(EPS, 1.0/3.0);
		default:
			throw new RuntimeException("only implemented for TWO_POINT, COMPLEX_STEP and THREE_POINT");
		}
	}

	/**
	 * Computes an absolute step from a relative step for finite difference calculation.
	 *
	 * `h` will always be np.float64.
	 * However, if `x0` or `f0` are smaller floating point dtypes (e.g. np.float32),
	 * then the absolute step size will be calculated from the smallest floating point size.
	 *
	 * @param relStep Relative step for the finite difference calculation
	 * @param x0 Parameter vector
	 * @param f0 function value (?)
	 * @param method {'2-point', '3-point', 'cs'}
	 * @return The absolute step size
	 */
	public static Matrix computeAbsoluteStep(Matrix relStep, Matrix x0, Matrix f0, FiniteDifferenceMethod method) {

		// this is used instead of np.sign(x0) because we need
	    // sign_x0 to be 1 when x0 == 0.
		Matrix x0Sign = x0.ge(0).toIntMatrix().times(2.0).minus(1.0);

		// This port stores all matrices as double; no need to dispatch on value type.
		final Class<?> x0Type = double.class;
		final Class<?> f0Type = double.class;

		double rStep = epsForMethod(x0Type, f0Type, method);

		final Matrix absStep;
		if (relStep == null) {
			absStep = Matrix.Factory.zeros(x0Sign.getSize());
			for (long[] pos: x0Sign.allCoordinates()) {
				absStep.setAsDouble(rStep * x0Sign.getAsInt(pos) * Math.max(1.0, Math.abs(x0.getAsDouble(pos))), pos);
			}
		} else {
			// User has requested specific relative steps.
	        // Don't multiply by max(1, abs(x0) because if x0 < 1 then their
			// requested step is not used.
			absStep = relStep.times(x0Sign).times(x0.abs());

			// however we don't want an abs_step of 0, which can happen if
	        // rel_step is 0, or x0 is 0. Instead, substitute a realistic step.
			for (long[] pos: absStep.allCoordinates()) {
				double x0Val = x0.getAsDouble(pos);
				double absStepVal = absStep.getAsDouble(pos);
				double dx = (x0Val + absStepVal) - x0Val;
				if (dx == 0.0) {
					double absStepToUse = rStep * x0Sign.getAsInt(pos) * Math.max(1.0, Math.abs(x0.getAsDouble(pos)));
					absStep.setAsDouble(absStepToUse, pos);
				}
			}
		}

		return absStep;
	}

	/**
	 * Finite-difference gradient of a scalar function of a scalar argument.
	 *
	 * Internally, this uses approxDerivative for vector-valued functions.
	 *
	 * @param f
	 * @param x0
	 * @param f0
	 * @param options
	 * @return
	 */
	public static double approxDerivative(DoubleUnaryOperator f, double x0, FiniteDifferenceOptions options) {
	    UnaryOperator<Matrix> fVec = (Matrix t) -> {
			return Matrix.Factory.linkToArray(new double[] { f.applyAsDouble(t.doubleValue()) });
		};
		Matrix x0Vec = Matrix.Factory.linkToArray(new double[] { x0 });
		Matrix f0Vec = null;
		return approxDerivative(fVec, x0Vec, f0Vec, options).doubleValue();
	}

	/**
	 * Finite-difference gradient of a scalar function of a scalar argument.
	 *
	 * Internally, this uses approxDerivative for vector-valued functions.
	 *
	 * @param f
	 * @param x0
	 * @param f0
	 * @param options
	 * @return
	 */
	public static double approxDerivative(ToDoubleFunction<Matrix> f, double x0, FiniteDifferenceOptions options) {
		UnaryOperator<Matrix> fVec = (Matrix t) -> {
			return Matrix.Factory.linkToArray(new double[] { f.applyAsDouble(t) });
		};
		Matrix x0Vec = Matrix.Factory.linkToArray(new double[] { x0 });
		Matrix f0Vec = null;
		return approxDerivative(fVec, x0Vec, f0Vec, options).doubleValue();
	}

	/**
	 * Finite-difference gradient of a scalar function of a scalar argument.
	 *
	 * Internally, this uses approxDerivative for vector-valued functions.
	 *
	 * @param f
	 * @param x0
	 * @param f0
	 * @param options
	 * @return
	 */
	public static double approxDerivative(ToDoubleFunction<Matrix> f, double x0, double f0, FiniteDifferenceOptions options) {
		Function<Matrix, Matrix> fVec = (Matrix t) -> {
			return Matrix.Factory.linkToArray(new double[] { f.applyAsDouble(t) });
		};
		Matrix x0Vec = Matrix.Factory.linkToArray(new double[] { x0 });
		Matrix f0Vec = Matrix.Factory.linkToArray(new double[] { f0 });
		return approxDerivative(fVec, x0Vec, f0Vec, options).doubleValue();
	}


	/**
	 * Finite-difference gradient of a real-valued function.
	 *
	 * Internally, this uses approxDerivative for vector-valued functions.
	 *
	 * @param f
	 * @param x0
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(ToDoubleFunction<Matrix> f, Matrix x0, double f0, FiniteDifferenceOptions options) {
		Function<Matrix, Matrix> fVec = (Matrix t) -> {
			return Matrix.Factory.linkToArray(new double[] { f.applyAsDouble(t) });
		};
		Matrix f0Vec = Matrix.Factory.linkToArray(new double[] { f0 });
		return approxDerivative(fVec, x0, f0Vec, options);
	}

	/**
	 * Finite-difference approximation of the first-order derivative matrix of a vector-valued function.
	 *
	 * @param f
	 * @param x0
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(Function<Matrix, Matrix> f, Matrix x0, Matrix f0, FiniteDifferenceOptions options) {
		Object args = null;
		BiFunction<Matrix, Object, Matrix> fArg =  (Matrix t, Object _args) -> {
			return f.apply(t);
		};
		return approxDerivative(fArg, x0, f0, options, args);
	}

	/**
	 * Compute finite difference approximation of the derivatives of a vector-valued
	 * function.
	 *
	 * If a function maps from R^n to R^m, its derivatives form m-by-n matrix called
	 * the Jacobian, where an element (i, j) is a partial derivative of f[i] with
	 * respect to x[j].
	 *
	 * See Also
	 * -----
	 * check_derivative : Check correctness of a function computing derivatives.
     *
     * Notes
     * -----
     * If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
     * determined from the smallest floating point dtype of `x0` or `fun(x0)`,
     * ``np.finfo(x0.dtype).eps``, s=2 for '2-point' method and
     * s=3 for '3-point' method. Such relative step approximately minimizes a sum
     * of truncation and round-off errors, see [1]_. Relative steps are used by
     * default. However, absolute steps are used when ``abs_step is not None``.
     * If any of the absolute or relative steps produces an indistinguishable
     * difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
     * automatic step size is substituted for that particular entry.
     *
     * A finite difference scheme for '3-point' method is selected automatically.
     * The well-known central difference scheme is used for points sufficiently
     * far from the boundary, and 3-point forward or backward scheme is used for
     * points near the boundary. Both schemes have the second-order accuracy in
     * terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point
     * forward and backward difference schemes.
     *
     * For dense differencing when m=1 Jacobian is returned with a shape (n,),
     * on the other hand when n=1 Jacobian is returned with a shape (m, 1).
     * Our motivation is the following: a) It handles a case of gradient
     * computation (m=1) in a conventional way. b) It clearly separates these two
     * different cases. b) In all cases np.atleast_2d can be called to get 2-D
     * Jacobian with correct dimensions.
     *
     * References
     * ----------
     * .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific
     *        Computing. 3rd edition", sec. 5.7.
     *
     * .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
     *        sparse Jacobian matrices", Journal of the Institute of Mathematics
     *        and its Applications, 13 (1974), pp. 117-120.
     *
     * .. [3] B. Fornberg, "Generation of Finite Difference Formulas on
     *        Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.
	 *
	 * @param fun     Function of which to estimate the derivatives. The argument x
	 *                passed to this function is ndarray of shape (n,) (never a
	 *                scalar even if n=1). It must return 1-D array_like of shape
	 *                (m,) or a scalar.
	 * @param x0      Point at which to estimate the derivatives. Float will be
	 *                converted to a 1-D array.
	 * @param f0      If not None it is assumed to be equal to ``fun(x0)``, in this
	 *                case the ``fun(x0)`` is not called. Default is None.
	 * @param options
	 * @param args    Additional arguments passed to `fun`. Empty by default.
	 * @return Finite difference approximation of the Jacobian matrix. If
	 *         `as_linear_operator` is True returns a LinearOperator with shape (m,
	 *         n). Otherwise it returns a dense array or sparse matrix depending on
	 *         how `sparsity` is defined. If `sparsity` is None then a ndarray with
	 *         shape (m, n) is returned. If `sparsity` is not None returns a
	 *         csr_matrix with shape (m, n). For sparse matrices and linear
	 *         operators it is always returned as a 2-D structure, for ndarrays, if
	 *         m=1 it is returned as a 1-D gradient array with shape (n,).
	 */
	public static Matrix approxDerivative(BiFunction<Matrix, Object, Matrix> fun, Matrix x0, Matrix f0,
			FiniteDifferenceOptions options, Object args) {

		switch(options.method()) {
		case TWO_POINT: // fall-through
		case THREE_POINT: // fall-through
		case COMPLEX_STEP:
			// ok
			break;
		default:
			throw new RuntimeException("Can only use TWO_POINT, THREE_POINT or COMPLEX_STEP");
		}

		if (x0.getSize().length != 2 || x0.getColumnCount() != 1) {
			throw new RuntimeException("x0 must be of shape (n,1)");
		}

		// TODO: prepareBounds:
		// make default infinite bounds if none specified

		if (    options.bounds().lb().getRowCount()    != x0.getRowCount() ||
				options.bounds().lb().getColumnCount() != x0.getColumnCount() ||
				options.bounds().ub().getRowCount()    != x0.getRowCount() ||
				options.bounds().ub().getColumnCount() != x0.getColumnCount()) {
			throw new RuntimeException("Inconsistent shapes between bounds and `x0`.");
		}

		if (options.asLinearOperator()) {
			for (long[] pos: x0.allCoordinates()) {
				double l = options.bounds().lb().getAsDouble(pos);
				double u = options.bounds().ub().getAsDouble(pos);
				if (l != Double.NEGATIVE_INFINITY || u != Double.POSITIVE_INFINITY) {
					throw new RuntimeException("Bounds not supported when `as_linear_operator` is True.");
				}
			}
		}

		if (x0.lt(options.bounds().lb()).or(x0.gt(options.bounds().ub())).toIntMatrix().getValueSum() > 0) {
			throw new RuntimeException("`x0` violates bound constraints.");
		}

		UnaryOperator<Matrix> funWrapped = x -> fun.apply(x, args);

		if (f0 == null) {
			f0 = funWrapped.apply(x0);
		}

		if (options.asLinearOperator()) {
			final Matrix relStep;
			if (options.relStep() == null) {
				relStep = Matrix.Factory.ones(x0.getSize()).times(epsForMethod(type(x0), type(f0), options.method()));
			} else {
				relStep = options.relStep();
			}
			return (Matrix) linearOperatorDifference(funWrapped, x0, f0, relStep, options.method());
		} else {
			// by default we use rel_step
			final Matrix absStep;
			if (options.absStep() == null) {
				absStep = computeAbsoluteStep(options.relStep(), x0, f0, options.method());
			} else {
				// user specifies an absolute step
				Matrix x0Sign = x0.ge(0).toIntMatrix().times(2.0).minus(1.0);
				absStep = options.absStep();

				// Cannot have a zero step.
				// This might happen if x0 is very large or small.
				// In which case fall back to relative step.
				Matrix dx = (x0.plus(absStep)).minus(x0);
				for (long[] pos: absStep.allCoordinates()) {
					double dxVal = dx.getAsDouble(pos);
					if (dxVal == 0.0) {
						double newAbsStep = epsForMethod(type(x0), type(f0), options.method()) *
								x0Sign.getAsDouble(pos) * Math.max(1.0, Math.abs(x0.getAsDouble(pos)));
						absStep.setAsDouble(newAbsStep, pos);
					}
				}
			}

			final AdjustedDifferencingScheme ads;
			switch (options.method()) {
			case TWO_POINT:
				ads = adjustSchemeToBounds(x0, absStep, 1, FiniteDifferenceMethod.ONE_SIDED, options.bounds().lb(), options.bounds().ub());
				break;
			case THREE_POINT:
				ads = adjustSchemeToBounds(x0, absStep, 1, FiniteDifferenceMethod.TWO_SIDED, options.bounds().lb(), options.bounds().ub());
				break;
			case COMPLEX_STEP:
				// useOneSided = false
				throw new RuntimeException("not implemented yet");
			default:
				throw new RuntimeException("only TWO_POINT, THREE_POINT and COMPLEX_STEP are allowed");
			}

			if (options.sparsity() == null) {
				return denseDifference(funWrapped, x0, f0, absStep, ads.useOneSided(), options.method());
			} else {
				return sparseDifference(funWrapped, x0, f0, absStep, ads.useOneSided(), options.sparsity(), options.method());
			}
		}
	}

	private static Class<?> type(Matrix A) {
		// This port stores all matrices as double.
		return double.class;
	}

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

		long m = f0.getRowCount();
		long n = x0.getRowCount();

		Matrix Jt = Matrix.Factory.zeros(n, m);
		Matrix hVecs = Matrix.Factory.eye(n, n);
		for (long[] pos: absStep.allCoordinates()) {
			// set diagonal to absStep
			hVecs.setAsDouble(absStep.getAsDouble(pos), pos[0], pos[0]);
		}

		for (long[] pos: absStep.availableCoordinates()) {
			int i = (int) pos[0];
			Matrix hI = hVecs.subMatrix(0, i, n-1, i);
			final double dx;
			final Matrix df;
			switch (method) {
			case TWO_POINT:
				Matrix x = x0.plus(hI);
				dx = x.getAsDouble(i, 0) - x0.getAsDouble(i, 0);
				df = fun.apply(x).minus(f0);
				break;
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

			for (long[] p2: df.allCoordinates()) {
				double j = df.getAsDouble(p2) / dx;
				Jt.setAsDouble(j, pos[0], p2[0]);
			}
		}

		return Jt.transpose();
	}

	private static Matrix sparseDifference(Function<Matrix, Matrix> fun, Matrix x0, Matrix f0,
			Matrix absStep, boolean[] useOneSided, Sparsity sparsity, FiniteDifferenceMethod method) {

		long m = f0.getRowCount();
		long n = x0.getRowCount();

		Matrix J = SparseMatrix.Factory.zeros(m, n);

		int[] groups = sparsity.sparsityGroups();
		int nGroups = Arrays.stream(groups).max().getAsInt() + 1;
		for (int group = 0; group < nGroups; ++group) {
			// Perturb variables which are in the same group simultaneously.
			Matrix e = Matrix.Factory.zeros(n, 1);
			for (int i=0; i<groups.length; ++i) {
				if (groups[i] == group) {
					e.setAsInt(1, i, 0);
				}
			}

			Matrix h = absStep.times(e);

			if (method == FiniteDifferenceMethod.TWO_POINT) {
				Matrix x = x0.plus(h);
				Matrix dx = x.minus(x0);
				Matrix df = fun.apply(x).minus(f0);
				for (long[] pos: sparsity.A().availableCoordinates()) {
					if (sparsity.A().getAsDouble(pos) != 0.0 && e.getAsInt(pos[0], 0) != 0) {
						// current coordinate has non-zero Jacobian entry and it has been influenced by e
						// --> expect a non-zero Jacobian element here
						J.setAsDouble(df.getAsDouble(pos[0], 0) / dx.getAsDouble(pos[1], 0), pos);
					}
				}
			} else if (method == FiniteDifferenceMethod.THREE_POINT) {
				// Here we do conceptually the same but separate one-sided and two-sided schemes.
				Matrix x1 = Matrix.Factory.zeros(n, 1);
				Matrix x2 = Matrix.Factory.zeros(n, 1);
				Matrix dx = Matrix.Factory.zeros(n, 1);
				for (long[] pos: e.availableCoordinates()) {
					if (e.getAsInt(pos) != 0) {
						final double x1Val, x2Val;
						if (useOneSided[(int) pos[0]]) {
							// These are the ones where one-sided differences have to be used due to proximity to bounds.
							x1Val = x0.getAsDouble(pos) + 1.0 * h.getAsDouble(pos);
							x2Val = x0.getAsDouble(pos) + 2.0 * h.getAsDouble(pos);
						} else {
							// These are the others, where regular central differencing can be used
				            // because they are far away enough from the bounds.
							x1Val = x0.getAsDouble(pos) - h.getAsDouble(pos);
							x2Val = x0.getAsDouble(pos) + h.getAsDouble(pos);
						}
						x1.setAsDouble(x1Val, pos);
						x2.setAsDouble(x2Val, pos);
						dx.setAsDouble(x2Val - x1Val, pos);
					}
				}

				Matrix f1 = fun.apply(x1);
				Matrix f2 = fun.apply(x2);

				Matrix df = Matrix.Factory.zeros(m, 1);
				for (int j=0; j<m; ++j) {
					final double dfVal;
					if (useOneSided[j]) {
						dfVal = -3.0*f0.getAsDouble(j, 0) + 4.0 * f1.getAsDouble(j, 0) - f2.getAsDouble(j, 0);
					} else {
						dfVal = f2.getAsDouble(j, 0) - f1.getAsDouble(j, 0);
					}
					df.setAsDouble(dfVal, j, 0);
				}

				for (long[] pos: sparsity.A().availableCoordinates()) {
					if (sparsity.A().getAsDouble(pos) != 0.0 && e.getAsInt(pos[0], 0) != 0) {
						// current coordinate has non-zero Jacobian entry and it has been influenced by e
						// --> expect a non-zero Jacobian element here
						J.setAsDouble(df.getAsDouble(pos[0], 0) / dx.getAsDouble(pos[1], 0), pos);
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
