package de.labathome.optimization;

import org.netlib.util.intW;

import com.github.fommil.netlib.LAPACK;

import de.labathome.LinAlg;

/**
 * Solve the Equality-Constrained Quadratic Programming Problem:
 *
 * <pre>
 * min wrt. x: q(x) = 1/2 x^T G x + x^T c
 * s.t.         A x = b
 *
 * where
 * G is the symmetric (n x n) Hessian matrix,
 * c and x are vectors in R^n and
 * A is the (m x n) Jacobian of constraints (with m <= n).
 * n = number of parameters
 * m = number of constraints
 * </pre>
 *
 * @see Nocedal/Wright, Numerical Optimization (2006), chapter 16.1
 * @see https://antonior92.github.io/posts/2017/05/projected-CG/
 */
public class QuadraticProgrammingSubproblem {

	private static LAPACK lapack;

	/** number of parameters of the problem */
	protected int n;

	/** number of constraints of the problem */
	protected int m;

	/** [n][n] Hessian matrix of the EQP problem */
	protected double[][] H;

	/** [n] gradient of the quadratic objective function */
	protected double[] c;

	/** [m][n] Jacobian matrix of the EQP problem */
	protected double[][] A;

	/** [m] Right-hand side of the constraint equation * (-1) */
	protected double[] b;

	/** [n] solution: vector of parameters */
	protected double[] x;

	/** [m] solution: Lagrange multipliers for constriants */
	protected double[] lambda;

	/**
	 * Setup an equality-constrained quadratic programming problem.
	 *
	 * @param n number of parameters of the problem
	 * @param m number of constraints of the problem
	 * @param H [n][n] Hessian matrix of the EQP problem
	 * @param c [n] gradient of the quadratic objective function
	 * @param A [m][n] Jacobian matrix of the EQP problem
	 * @param b [m] Right-hand side of the constraint equation * (-1)
	 */
	public QuadraticProgrammingSubproblem(int n, int m, double[][] H, double[] c, double[][] A, double[] b) {
		this.n = n;
		this.m = m;
		this.H = H;
		this.c = c;
		this.A = A;
		this.b = b;

		x = new double[n];
		lambda = new double[m];
	}

	/**
	 * Get the solution vector.
	 *
	 * @return [n] values of parameters
	 */
	public double[] getX() {
		return x;
	}

	/**
	 * Get the vector of Lagrange multipliers for the constraints.
	 *
	 * @return [m] values of Lagrange multipliers
	 */
	public double[] getLambda() {
		return lambda;
	}

	/** solve by direct factorization of the KKT matrix (16.5) */
	public void directFactorization() {
		if (lapack == null) {
			lapack = LAPACK.getInstance();
		}

		// 1. build explicit KKT matrix:
		// [ G A^T ]
		// [ A  0  ]
		// It is stored in column-major format for compatibility with LAPACK.
		final double[] kkt = new double[(n+m)*(n+m)];

		// copy G into top left block of KKT matrix
		for (int iR = 0; iR < n; ++iR) {
			for (int iC=0; iC < n; ++iC) {
				kkt[iC * (n+m) + iR] = H[iR][iC];
			}
		}

		for (int j=0; j<m; ++j) {
			for (int i=0; i<n; ++i) {
				// copy A into bottom left block of KKT matrix
				kkt[i * (n+m) + (n+j)] = A[j][i];

				// copy A^T into top right block of KKT matrix
				kkt[(n+j) * (n+m) + i] = A[j][i];
			}
		}

		// 2. build RHS vector
		// [ -c ]
		// [ -b ]
		final double[] rhs = new double[n+m];
		for (int i=0; i<n; ++i) {
			rhs[i] = -c[i];
		}
		for (int j=0; j<m; ++j) {
			rhs[n+j] = -b[j]; // TODO: change to b --> also in API!
		}

		// TODO: Use a symmetric indefinite factorization
		//       to solve the system twice as fast (because of the symmetry).

		// 3. obtain LU factorization of KKT matrix using LAPACK's dgetrf
		final int[] ipiv = new int[n+m];
		intW info = new intW(0);
		lapack.dgetrf(n+m, n+m, kkt, n+m, ipiv, info);
		if (info.val != 0) {
			throw new RuntimeException(String.format("DGETRF returned info = %d", info.val));
		}

		// 4. solve using LAPACK's dgetrs
		String trans = "N";
		lapack.dgetrs(trans, n+m, 1, kkt, n+m, ipiv, rhs, n+m, info);
		if (info.val != 0) {
			throw new RuntimeException(String.format("DGETRS returned info = %d", info.val));
		}

		// 5. copy solution back into appropriate vectors
		for (int i = 0; i < n; ++i) {
			x[i] = rhs[i];
		}
		for (int j = 0; j < m; ++j) {
			lambda[j] = -rhs[n + j];
		}
	}

	/**
	 * Find the intersection between segment (or line) and spherical constraints.
	 *
	 * Find the intersection between the segment (or line) defined by the parametric
	 * equation {@code x(t) = z + t*d} and the ball {@code ||x|| <= trust_radius}.
	 *
	 * @param z           [n] initial point
	 * @param d           [n] direction
	 * @param trustRadius ball radius
	 * @return
	 */
	public static IntersectionResult sphereIntersections(double[] z, double[] d, double trustRadius) {
		boolean entireLine = false;
		return sphereIntersections(z, d, trustRadius, entireLine);
	}

	/**
	 * Find the intersection between segment (or line) and spherical constraints.
	 *
	 * Find the intersection between the segment (or line) defined by the parametric
	 * equation {@code x(t) = z + t*d} and the ball {@code ||x|| <= trust_radius}.
	 *
	 * @param z           [n] initial point
	 * @param d           [n] direction
	 * @param trustRadius ball radius
	 * @param entireLine  When {@code true}, the function returns the intersection
	 *                    between the line {@code x(t) = z + t*d} ({@code t} can
	 *                    assume any value) and the ball {@code ||x|| <= trust_radius}.
	 *                    When {@code false}, the function returns the intersection
	 *                    between the segment {@code x(t) = z + t*d}, {@code 0 <= t <= 1},
	 *                    and the ball.
	 */
	public static IntersectionResult sphereIntersections(double[] z, double[] d, double trustRadius, boolean entireLine) {

		// special case when d == 0
		if (LinAlg.norm(d) == 0.0) {
			return new IntersectionResult(0.0, 0.0, false);
		}

		// check for infinite trust radius
		if (Double.isInfinite(trustRadius)) {
			final double tA, tB;
			if (entireLine) {
				tA = Double.NEGATIVE_INFINITY;
				tB = Double.POSITIVE_INFINITY;
			} else {
				tA = 0.0;
				tB = 1.0;
			}
			return new IntersectionResult(tA, tB, true);
		}

		double a = LinAlg.dot(d, d);
		double b = 2.0 * LinAlg.dot(z, d);
		double c = LinAlg.dot(z, z) - trustRadius * trustRadius;
		double discriminant = b * b - 4 * a * c;
		if (discriminant < 0.0) {
			// line does not hit the ball (?)
			return new IntersectionResult(0.0, 0.0, false);
		}

		double sqrtDiscriminant = Math.sqrt(discriminant);

		// The following calculation is mathematically equivalent to:
	    // ta = (-b - sqrt_discriminant) / (2*a)
	    // tb = (-b + sqrt_discriminant) / (2*a)
	    // but produce smaller round off errors.
	    // Look at Matrix Computation p.97 for a better justification.
		double aux = b + Math.copySign(sqrtDiscriminant, b);
		double tA = -aux / (2.0 * a);
		double tB = -2.0 * c / aux;

		// ta, tb = sorted([ta, tb])
		if (tB < tA) {
			double temp = tB;
			tB = tA;
			tA = temp;
		}

		final boolean intersect;
		if (entireLine) {
			intersect = true;
		} else {
			// Checks to see if intersection happens within vectors length.
			if (tB < 0.0 || tA > 1.0) {
				intersect = false;
				tA = 0.0;
				tB = 0.0;
			} else {
				intersect = true;
				// Restrict intersection interval between 0 and 1.
				tA = Math.max(0.0, tA);
				tB = Math.min(1.0, tB);
			}
		}

		return new IntersectionResult(tA, tB, intersect);
	}

	/**
	 * Find the intersection between segment (or line) and box constraints.
	 *
	 * Find the intersection between the segment (or line) defined by the
	 * parametric  equation {@code x(t) = z + t*d} and the rectangular box
	 * {@code lb <= x <= ub}.
	 *
	 * @param zIn [n] initial point
	 * @param dIn [n] direction
	 * @param lbIn [n] lower bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @param ubIn [n] upper bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @return
	 */
	public static IntersectionResult boxIntersections(double[] z, double[] d, double[] lb, double[] ub) {
		boolean entireLine = false;
		return boxIntersections(z, d, lb, ub, entireLine);
	}

	/**
	 * Find the intersection between segment (or line) and box constraints.
	 *
	 * Find the intersection between the segment (or line) defined by the
	 * parametric  equation {@code x(t) = z + t*d} and the rectangular box
	 * {@code lb <= x <= ub}.
	 *
	 * @param zIn [n] initial point
	 * @param dIn [n] direction
	 * @param lbIn [n] lower bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @param ubIn [n] upper bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @param entireLine When {@code true}, the function returns the intersection between the line
	 *                   {@code x(t) = z + t*d} ({@code t} can assume any value) and the rectangular box.
	 *                   When {@code false}, the function returns the intersection between the segment
	 *                   {@code x(t) = z + t*d}, {@code 0 <= t <= 1}, and the rectangular box.
	 * @return
	 */
	public static IntersectionResult boxIntersections(double[] zIn, double[] dIn, double[] lbIn, double[] ubIn, boolean entireLine) {

		// special case when d == 0
		if (LinAlg.norm(dIn) == 0.0) {
			return new IntersectionResult(0.0, 0.0, false);
		}

		final int nIn = zIn.length;

		int n = 0;
		for (int i=0; i<nIn; ++i) {
			if (dIn[i] == 0.0) {
				// If the boundaries are not satisfied for some coordinate
				// for which "d" is zero, there is no box-line intersection.
				if (zIn[i] < lbIn[i] || zIn[i] > ubIn[i]) {
					return new IntersectionResult(0.0, 0.0, false);
				}
			} else {
				n++;
			}
		}

		// remove values for which d is zero
		double[] z = new double[n];
		double[] d = new double[n];
		double[] lb = new double[n];
		double[] ub = new double[n];
		int idx = 0;
		for (int i=0; i<nIn; ++i) {
			if (dIn[i] != 0.0) {
				z[idx] = zIn[i];
				d[idx] = dIn[i];
				lb[idx] = lbIn[i];
				ub[idx] = ubIn[i];
				idx++;
			}
		}

		// Find a series of intervals (t_lb[i], t_ub[i]).
		// Get the intersection of all those intervals.
		double tA = Double.NEGATIVE_INFINITY;
		double tB = Double.POSITIVE_INFINITY;
		for (int i=0; i<n; ++i) {
			double t_lb = (lb[i] - z[i]) / d[i];
			double t_ub = (ub[i] - z[i]) / d[i];
			double minT = Math.min(t_lb, t_ub);
			double maxT = Math.max(t_lb, t_ub);
			tA = Math.max(tA, minT);
			tB = Math.min(tB, maxT);
		}

		// check if intersection is feasible
		final boolean intersect = (tA <= tB);

		// Checks to see if intersection happens within vectors length.
		if (!entireLine) {
			if (tB < 0.0 || tA > 1.0) {
				return new IntersectionResult(0.0, 0.0, false);
			} else {
				// Restrict intersection interval between 0 and 1.
				tA = Math.max(0.0, tA);
				tB = Math.min(1.0, tB);
			}
		}

		return new IntersectionResult(tA, tB, intersect);
	}

	public static IntersectionResult boxSphereIntersections(double[] z, double[] d, double[] lb, double[] ub, double trustRadius) {
		boolean entireLine = false;
		return boxSphereIntersections(z, d, lb, ub, trustRadius, entireLine);
	}

	public static IntersectionResult boxSphereIntersections(double[] z, double[] d, double[] lb, double[] ub, double trustRadius, boolean entireLine) {
		return boxSphereIntersectionsWithExtraInfo(z, d, lb, ub, trustRadius, entireLine)[0];
	}

	public static IntersectionResult[] boxSphereIntersectionsWithExtraInfo(double[] z, double[] d, double[] lb, double[] ub, double trustRadius) {
		boolean entireLine = false;
		return boxSphereIntersectionsWithExtraInfo(z, d, lb, ub, trustRadius, entireLine);
	}

	/**
	 * Find the intersection between segment (or line) and box/sphere constraints.
	 *
	 * Find the intersection between the segment (or line) defined by the
	 * parametric  equation {@code x(t) = z + t*d}, the rectangular box
	 * {@code lb <= x <= ub} and the ball {@code ||x|| <= trust_radius}.
	 *
	 * @param z [n] initial point
	 * @param d [n] direction
	 * @param lb [n] lower bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @param ub [n] upper bounds to each one of the components of {@code x},
	 *               used to delimit the rectangular box
	 * @param trustRadius ball radius
	 * @param entireLine When {@code true}, the function returns the intersection between the line
	 *                   {@code x(t) = z + t*d} ({@code t} can assume any value) and the constraints.
	 *                   When {@code false}, the function returns the intersection between the segment
	 *                   {@code x(t) = z + t*d}, {@code 0 <= t <= 1} and the constraints.
	 * @return [3] The  first element is the combined intersection result.
	 *             The second element is the intersection result from {@code sphereIntersections}.
	 *             The  third element is the intersection result from {@code boxIntersections}.
	 */
	public static IntersectionResult[] boxSphereIntersectionsWithExtraInfo(double[] z, double[] d, double[] lb, double[] ub, double trustRadius, boolean entireLine) {

		IntersectionResult rS = sphereIntersections(z, d, trustRadius, entireLine);
		IntersectionResult rB = boxIntersections(z, d, lb, ub, entireLine);

		double tA = Math.max(rS.tA(), rB.tA());
		double tB = Math.min(rS.tB(), rB.tB());
		boolean intersect = rS.intersect() && rB.intersect() && (tA <= tB);

		return new IntersectionResult[] {
				new IntersectionResult(tA, tB, intersect),
				rS,
				rB
		};
	}

	/**
	 * Approximately  minimize {@code 1/2*|| A x + b ||^2} inside trust-region.
	 *
	 * Approximately solve the problem of minimizing {@code 1/2*|| A x + b ||^2}
	 * subject to {@code ||x|| < Delta} and {@code lb <= x <= ub} using a modification
	 * of the classical dogleg approach.
	 *
	 * Based on implementations described in pp. 885-886 from [1].
	 *
	 * [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
	 *     "An interior point algorithm for large-scale nonlinear
	 *     programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
	 *
	 * @param A [m][n] Matrix {@code A} in the minimization problem.
	 *                 It should have dimensions {@code (m, n)} such that {@code m < n}.
	 * @param Y [n][m] LinearOperator that apply the projection matrix
	 *                 {@code Q = A.T inv(A A.T)} to the vector. The obtained vector
	 *                 {@code y = Q x} being the minimum norm solution of {@code A y = x}.
	 * @param b [m] Vector {@code b}in the minimization problem.
	 * @param trustRadius Trust radius to be considered. Delimits a sphere boundary to the problem.
	 * @param lb [n] Lower bounds to each one of the components of {@code x}.
	 *               It is expected that {@code lb <= 0}, otherwise the algorithm
	 *               may fail. If {@code lb[i] = Double.NEGATIVE_INFINITY}, the lower
	 *               bound for the i-th component is just ignored.
	 * @param ub [n] Upper bounds to each one of the components of {@code x}.
	 *               It is expected that {@code ub >= 0}, otherwise the algorithm
	 *               may fail. If {@code ub[i] = Double.POSITIVE_INFINITY}, the upper bound for the i-th
	 *               component is just ignored.
	 * @return [n] Solution to the problem.
	 */
	public static double[] modifiedDogleg(double[][] A, double[][] Y, double[] b, double trustRadius, double[] lb, double[] ub) {

		// Compute minimum norm minimizer of 1/2*|| A x + b ||^2.
		double[] newtonPoint = LinAlg.dot(A, b, -1.0);

		if (insideBoxBoundaries(newtonPoint, lb, ub) && LinAlg.norm(newtonPoint) <= trustRadius) {
			return newtonPoint;
		}

		// Compute gradient vector {@code g = A.T b}
		double[] g = LinAlg.dot(A, true, b);

		// Compute Cauchy point:
		// {@code cauchy_point = g.T g / (g.T A.T A g)}
		double[] A_g = LinAlg.dot(A, g);
		double cauchyScale = -LinAlg.dot(g, g) / LinAlg.dot(A_g, A_g);
		double[] cauchyPoint = LinAlg.mulElem(g, cauchyScale);

		// Origin
		double[] origin = new double[cauchyPoint.length];

		// Check the segment between cauchy_point and newton_point for a possible solution.
		double[] z = cauchyPoint;
		double[] p = LinAlg.subtract(newtonPoint, cauchyPoint);
		IntersectionResult r1 = boxSphereIntersections(z, p, lb, ub, trustRadius);
		double alpha = r1.tB();

		final double[] x1;
		if (!r1.intersect()) {
			// Check the segment between the origin and cauchy_point for a possible solution.
			z = origin;
			p = cauchyPoint;
			IntersectionResult r2 = boxSphereIntersections(z, p, lb, ub, trustRadius);
			alpha = r2.tB();
		}
		x1 = LinAlg.add(z, LinAlg.mulElem(p, alpha));

		// Check the segment between origin and newton_point for a possible solution.
		z = origin;
		p = newtonPoint;
		IntersectionResult r3 = boxSphereIntersections(z, p, lb, ub, trustRadius);
		alpha = r3.tB();
		double[] x2 = LinAlg.add(z, LinAlg.mulElem(p, alpha));

		// Return the best solution among x1 and x2.
		double norm1 = LinAlg.norm(LinAlg.add(LinAlg.dot(A, x1), b));
		double norm2 = LinAlg.norm(LinAlg.add(LinAlg.dot(A, x2), b));
		if (norm1 < norm2) {
			return x1;
		} else {
			return x2;
		}
	}









	/**
	 * Return clipped value of x.
	 * @param x  [n] position vector to force into bounds
	 * @param lb [n] lower bounds
	 * @param ub [n] upper bounds
	 * @return coerced copy of x such that lb <= x <= ub for all entries
	 */
	public static double[] reinforceBoxBoundaries(double[] x, double[] lb, double[] ub) {
		double[] clippedX = x.clone();
		for (int i=0; i<x.length; ++i) {
			double temp = Math.max(x[i], lb[i]);
			clippedX[i] = Math.min(temp, ub[i]);
		}
		return clippedX;
	}

	/**
	 * Check if lb <= x <= ub.
	 *
	 * @param x  [n] position to test
	 * @param lb [n] lower bounds
	 * @param ub [n] upper bounds
	 * @return true of lb <= x <= ub for all entries, false otherwise
	 */
	public static boolean insideBoxBoundaries(double[] x, double[] lb, double[] ub) {
		for (int i=0; i<x.length; ++i) {
			if (x[i] < lb[i] || x[i] > ub[i]) {
				return false;
			}
		}
		return true;
	}
}
