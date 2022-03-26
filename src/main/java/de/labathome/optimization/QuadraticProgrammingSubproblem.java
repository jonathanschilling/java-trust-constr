package de.labathome.optimization;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DenseDoubleMatrix2D;
import org.ujmp.core.doublematrix.DoubleMatrix;
import org.ujmp.core.doublematrix.SparseDoubleMatrix;
import org.ujmp.core.doublematrix.calculation.general.decomposition.LU.LUMatrix;

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

	/** number of parameters of the problem */
	protected int n;

	/** number of constraints of the problem */
	protected int m;

	/** [n][n] Hessian matrix of the EQP problem */
	protected Matrix H;

	/** [n] gradient of the quadratic objective function */
	protected Matrix c;

	/** [m][n] Jacobian matrix of the EQP problem */
	protected Matrix A;

	/** [m] Right-hand side of the constraint equation * (-1) */
	protected Matrix b;

	/** [n] solution: vector of parameters */
	protected DenseDoubleMatrix2D x;

	/** [m] solution: Lagrange multipliers for constriants */
	protected DenseDoubleMatrix2D lambda;

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
	public QuadraticProgrammingSubproblem(int n, int m, Matrix H, Matrix c, Matrix A, Matrix b) {
		this.n = n;
		this.m = m;
		this.H = H;
		this.c = c;
		this.A = A;
		this.b = b;

		x = DenseMatrix.Factory.zeros(n, 1);
		lambda = DenseMatrix.Factory.zeros(m, 1);
	}

	/**
	 * Get the solution vector.
	 *
	 * @return [n] values of parameters
	 */
	public DenseDoubleMatrix2D getX() {
		return x;
	}

	/**
	 * Get the vector of Lagrange multipliers for the constraints.
	 *
	 * @return [m] values of Lagrange multipliers
	 */
	public DenseDoubleMatrix2D getLambda() {
		return lambda;
	}

	/** solve by direct factorization of the KKT matrix (16.5) */
	public void directFactorization() {

		// 1. build explicit KKT matrix:
		// [ G A^T ]
		// [ A  0  ]
		Matrix kkt = SparseDoubleMatrix.Factory.zeros(n+m, n+m);

		// copy G into top left block of KKT matrix
		for (long[] pos: H.availableCoordinates()) {
			kkt.setAsDouble(H.getAsDouble(pos), pos);
		}

		for (long[] pos: A.availableCoordinates()) {
			double aVal = A.getAsDouble(pos);

			// copy A into bottom left block of KKT matrix
			kkt.setAsDouble(aVal, n+pos[0], pos[1]);

			// copy A^T into top right block of KKT matrix
			kkt.setAsDouble(aVal, pos[1], n+pos[0]);
		}

		// 2. build RHS vector
		// [ -c ]
		// [ -b ]
		Matrix rhs = DoubleMatrix.Factory.zeros(n+m, 1);
		for (long[] pos: c.availableCoordinates()) {
			rhs.setAsDouble(-c.getAsDouble(pos), pos);
		}
		for (long[] pos: b.availableCoordinates()) {
			// TODO: change to b for consistency with book --> also in API!
			rhs.setAsDouble(-b.getAsDouble(pos), n + pos[0], pos[1]);
		}

		// TODO: Use a symmetric indefinite factorization
		//       to solve the system twice as fast (because of the symmetry).

		// 3. obtain LU factorization of KKT matrix
		LUMatrix lu = new LUMatrix(kkt);

		// 4. solve
		Matrix sln = lu.solve(rhs);

		// 5. copy solution back into appropriate vectors
		for (long[] pos: x.allCoordinates()) {
			x.setAsDouble(sln.getAsDouble(pos), pos);
		}
		for (long[] pos: lambda.allCoordinates()) {
			lambda.setAsDouble(-sln.getAsDouble(n + pos[0], pos[1]), pos);
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
	public static Matrix modifiedDogleg(Matrix A, Matrix Y, Matrix b, double trustRadius, double[] lb, double[] ub) {

		// Compute minimum norm minimizer of 1/2*|| A x + b ||^2.
		Matrix newtonPoint = Y.mtimes(b).times(-1);

		if (insideBoxBoundaries(newtonPoint.transpose().toDoubleArray()[0], lb, ub)
				&& newtonPoint.norm2() <= trustRadius) {
			return newtonPoint;
		}

		// Compute gradient vector {@code g = A.T b}
		Matrix g = A.transpose().mtimes(b);

		// Compute Cauchy point:
		// {@code cauchy_point = g.T g / (g.T A.T A g)}
		Matrix A_g = A.mtimes(g);
		double cauchyScale = -g.transpose().mtimes(g).doubleValue() / A_g.transpose().mtimes(A_g).doubleValue();
		Matrix cauchyPoint = g.times(cauchyScale);

		// Origin
		Matrix origin = Matrix.Factory.zeros(cauchyPoint.getRowCount(), cauchyPoint.getColumnCount());

		// Check the segment between cauchy_point and newton_point for a possible solution.
		Matrix z = cauchyPoint;
		Matrix p = newtonPoint.minus(cauchyPoint);
		IntersectionResult r1 = boxSphereIntersections(
				z.transpose().toDoubleArray()[0],
				p.transpose().toDoubleArray()[0],
				lb, ub, trustRadius);
		double alpha = r1.tB();

		if (!r1.intersect()) {
			// Check the segment between the origin and cauchy_point for a possible solution.
			z = origin;
			p = cauchyPoint;
			IntersectionResult r2 = boxSphereIntersections(
					z.transpose().toDoubleArray()[0],
					p.transpose().toDoubleArray()[0],
					lb, ub, trustRadius);
			alpha = r2.tB();
		}
		Matrix x1 = z.plus(p.times(alpha));

		// Check the segment between origin and newton_point for a possible solution.
		z = origin;
		p = newtonPoint;
		IntersectionResult r3 = boxSphereIntersections(
				z.transpose().toDoubleArray()[0],
				p.transpose().toDoubleArray()[0],
				lb, ub, trustRadius);
		alpha = r3.tB();
		Matrix x2 = z.plus(p.times(alpha));

		// Return the best solution among x1 and x2.
		double norm1 = A.mtimes(x1).plus(b).norm2();
		double norm2 = A.mtimes(x2).plus(b).norm2();
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
