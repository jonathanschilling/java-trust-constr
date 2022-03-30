package org.scipy.optimize.minimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class NumDiff {

	/**
	 * Finite-difference gradient of a real-valued function.
	 *
	 * Internally, this uses approxDerivative for vector-valued functions.
	 *
	 * @param f
	 * @param x
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(ToDoubleFunction<Matrix> f, Matrix x, double f0, FiniteDifferenceOptions options) {

		Function<Matrix, Matrix> fVec = (Matrix t) -> {
			return Matrix.Factory.linkToArray(new double[] { f.applyAsDouble(t) });
		};

		Matrix f0Vec = Matrix.Factory.linkToArray(new double[] { f0 });

		return approxDerivative(fVec, x, f0Vec, options);
	}

	/**
	 * Finite-difference approximation of the first-order derivative matrix of a vector-valued function.
	 *
	 * @param f
	 * @param x
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(Function<Matrix, Matrix> f, Matrix x, Matrix f0, FiniteDifferenceOptions options) {
		Object args = null;
		BiFunction<Matrix, Object, Matrix> fArg =  (Matrix t, Object _args) -> {
			return f.apply(t);
		};
		return approxDerivative(fArg, x, f0, options, args);
	}

	public static Matrix approxDerivative(BiFunction<Matrix, Object, Matrix> f, Matrix x, Matrix f0, FiniteDifferenceOptions options, Object args) {

		// TODO

		return null;
	}

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
		for (long[] srcPos: A.availableCoordinates()) {
			for (long[] tgtPos: orderedA.allCoordinates()) {
				//if (srcPos[0] == tgtPos[0] && order[(int) srcPos[1]] == tgtPos[1]) {
				if (srcPos[0] == tgtPos[0] && srcPos[1] == order[(int) tgtPos[1]]) {
					System.out.printf("(%d,%d) --> (%d,%d)\n", srcPos[0], srcPos[1], tgtPos[0], tgtPos[1]);
					orderedA.setAsDouble(A.getAsDouble(srcPos), tgtPos);
				}
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
			Matrix aCol = A.selectColumns(Ret.LINK, i);
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
					Matrix aOtherCol = A.selectColumns(Ret.LINK, j);
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
			System.out.println("handle col i = " + i);
			if (groups[i] >= 0) {
				// A group was already assigned.
				continue;
			}

			groups[i] = currentGroup;
			boolean allGrouped = true;

			// Here we store the union of grouped columns.
			Arrays.fill(union, 0);
			Matrix ithCol = A.subMatrix(Ret.LINK, 0, i, A.getRowCount()-1, i);
			for (long[] pos: ithCol.availableCoordinates()) {
				if (ithCol.getAsDouble(pos) != 0.0) {
					System.out.println("  k = " + pos[0]);
					union[(int) pos[0]] = 1;
				}
			}
			System.out.println("  union = " + Arrays.toString(union));

			for (int j = 0; j < n; ++j) {
				if (groups[j] < 0) {
					allGrouped = false;
				} else {
					continue;
				}

				// Determine if j-th column intersects with the union.
				boolean intersect = false;
				Matrix jthCol = A.subMatrix(Ret.LINK, 0, j, A.getRowCount()-1, j);
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
							System.out.println("  k = " + pos[0]);
							union[(int) pos[0]] = 1;
						}
					}
					System.out.println("  union = " + Arrays.toString(union));
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
}
