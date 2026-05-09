package org.scipy.optimize.minimize;

import java.util.Arrays;
import java.util.LinkedList;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.records.CGInfo;
import org.scipy.optimize.minimize.records.IntersectionResult;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.DenseSolve;
import org.scipy.optimize.minimize.sparse.SparseAssembly;

import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.MatrixOps;

/**
 * Quadratic-programming subproblem helpers used inside the trust-region
 * SQP / IP loops: the projected-CG inner solver, the equality-constrained
 * KKT direct factorization, the modified-dogleg trust-region step, and the
 * segment/box/sphere intersection routines that drive trust-region step
 * acceptance.
 */
public class QPSubproblem {

	/**
	 * Solve the equality-constrained quadratic program
	 * <pre>
	 *   minimize  q(x) = 1/2 x^T G x + x^T c
	 *   subject to    A x = b
	 * </pre>
	 * where {@code G} is the symmetric {@code n x n} Hessian, {@code c, x}
	 * are length-{@code n} vectors, and {@code A} is the {@code m x n}
	 * Jacobian of constraints (with {@code m &le; n}).
	 *
	 * <p>See Nocedal &amp; Wright, <i>Numerical Optimization</i>, 2nd ed.
	 * (2006), chapter 16.1.
	 *
	 * @param H {@code n x n} Hessian matrix of the EQP problem
	 * @param c length-{@code n} gradient of the quadratic objective function
	 * @param A {@code m x n} Jacobian matrix of the equality constraints
	 * @param b length-{@code m} right-hand side of the constraint equation, sign-flipped
	 * @return two-element array {@code [x, lambda]} containing the solution and Lagrange multipliers
	 */
	public static Matrix[] eqpKktFact(Matrix H, Matrix c, Matrix A, Matrix b) {

		int n = (int) H.getRowCount();
		int m = (int) A.getRowCount();

		// 1. build explicit KKT matrix in CSR form using sparse-aware block assembly:
		// [ G A^T ]
		// [ A  0  ]
		CSRMatrix hCsr = CSRMatrix.fromMatrix(H);
		CSRMatrix aCsr = CSRMatrix.fromMatrix(A);
		CSRMatrix aTCsr = aCsr.transpose().toCSR();
		CSRMatrix kkt = SparseAssembly.blockArray(new CSRMatrix[][] {
				{ hCsr, aTCsr },
				{ aCsr, null  },
		});

		// 2. build RHS vector [ -c; -b ]
		double[] rhs = new double[n + m];
		for (int i = 0; i < n; ++i) {
			rhs[i] = -c.getAsDouble(i, 0);
		}
		for (int i = 0; i < m; ++i) {
			rhs[n + i] = -b.getAsDouble(i, 0);
		}

		// 3. solve via dense LU on the assembled KKT.
		// TODO: When the project ships a sparse direct solver this is the only call site
		//       that needs to switch -- the KKT matrix above is already CSR.
		// TODO: Use a symmetric indefinite factorization
		//       to solve the system twice as fast (because of the symmetry).
		double[] sln = DenseSolve.solveLU(kkt.toDense(), rhs);

		// 4. copy solution back into column vectors.
		Matrix x = Matrix.Factory.zeros(n, 1);
		for (int i = 0; i < n; ++i) {
			x.setAsDouble(sln[i], i, 0);
		}
		Matrix lambda = Matrix.Factory.zeros(m, 1);
		for (int i = 0; i < m; ++i) {
			lambda.setAsDouble(-sln[n + i], i, 0);
		}

		return new Matrix[] {x, lambda};
	}

	/**
	 * Find the intersection between segment (or line) and spherical constraints.
	 * <p>
	 * Find the intersection between the segment (or line) defined by the parametric
	 * equation {@code x(t) = z + t*d} and the ball {@code ||x|| <= trust_radius}.
	 *
	 * @param z           length-{@code n} initial point
	 * @param d           length-{@code n} direction
	 * @param trustRadius ball radius
	 * @return the {@code t}-interval over which the segment lies inside the
	 *         ball, or {@code intersect=false} if it never enters
	 */
	public static IntersectionResult sphereIntersections(double[] z, double[] d, double trustRadius) {
		boolean entireLine = false;
		return sphereIntersections(z, d, trustRadius, entireLine);
	}

	/**
	 * Find the intersection of a segment (or line) with the ball
	 * {@code ||x|| &le; trustRadius}, where the segment is parameterised as
	 * {@code x(t) = z + t*d}.
	 *
	 * @param z           length-{@code n} initial point
	 * @param d           length-{@code n} direction
	 * @param trustRadius ball radius
	 * @param entireLine  if {@code true}, allow {@code t} to range over all
	 *                    of {@code R}; if {@code false}, restrict
	 *                    {@code 0 &le; t &le; 1}
	 * @return the {@code t}-interval over which the segment lies inside the
	 *         ball, or {@code intersect=false} if it never enters
	 */
	public static IntersectionResult sphereIntersections(double[] z, double[] d, double trustRadius, boolean entireLine) {

		// special case when d == 0
		if (MatrixOps.norm2(d) == 0.0) {
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

		double a = MatrixOps.dot(d, d);
		double b = 2.0 * MatrixOps.dot(z, d);
		double c = MatrixOps.dot(z, z) - trustRadius * trustRadius;
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
	 * Find the intersection of a segment (or line) with the rectangular box
	 * {@code lb &le; x &le; ub}, where the segment is parameterised as
	 * {@code x(t) = z + t*d}. Convenience overload that disables the
	 * &quot;entire line&quot; mode.
	 *
	 * @param z  length-{@code n} initial point
	 * @param d  length-{@code n} direction
	 * @param lb length-{@code n} lower bounds of the box
	 * @param ub length-{@code n} upper bounds of the box
	 * @return {@code (tA, tB, intersect)} -- the {@code t}-interval over which
	 *         the segment lies inside the box, or {@code intersect=false}.
	 */
	public static IntersectionResult boxIntersections(double[] z, double[] d, double[] lb, double[] ub) {
		boolean entireLine = false;
		return boxIntersections(z, d, lb, ub, entireLine);
	}

	/**
	 * Find the intersection of a segment (or line) with the rectangular box
	 * {@code lb &le; x &le; ub}, where the segment is parameterised as
	 * {@code x(t) = z + t*d}.
	 *
	 * @param zIn  length-{@code n} initial point
	 * @param dIn  length-{@code n} direction
	 * @param lbIn length-{@code n} lower bounds of the box
	 * @param ubIn length-{@code n} upper bounds of the box
	 * @param entireLine if {@code true}, allow {@code t} to range over all
	 *                   of {@code R}; if {@code false}, restrict
	 *                   {@code 0 &le; t &le; 1}
	 * @return the {@code t}-interval over which the segment lies inside the
	 *         box, or {@code intersect=false} if it never enters
	 */
	public static IntersectionResult boxIntersections(double[] zIn, double[] dIn, double[] lbIn, double[] ubIn, boolean entireLine) {

		// special case when d == 0
		if (MatrixOps.norm2(dIn) == 0.0) {
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

	/**
	 * Convenience overload of
	 * {@link #boxSphereIntersections(double[], double[], double[], double[], double, boolean)}
	 * with {@code entireLine = false}.
	 *
	 * @param z           initial point
	 * @param d           direction
	 * @param lb          box lower bounds
	 * @param ub          box upper bounds
	 * @param trustRadius sphere radius
	 * @return the intersection {@code t}-interval
	 */
	public static IntersectionResult boxSphereIntersections(double[] z, double[] d, double[] lb, double[] ub, double trustRadius) {
		boolean entireLine = false;
		return boxSphereIntersections(z, d, lb, ub, trustRadius, entireLine);
	}

	/**
	 * Find the intersection of a segment (or line) with both a box
	 * {@code lb &le; x &le; ub} and a ball {@code ||x|| &le; trustRadius}.
	 *
	 * @param z           initial point
	 * @param d           direction
	 * @param lb          box lower bounds
	 * @param ub          box upper bounds
	 * @param trustRadius sphere radius
	 * @param entireLine  if {@code true}, allow {@code t} to range over all
	 *                    of {@code R}; if {@code false}, restrict to the segment
	 * @return the intersection {@code t}-interval
	 */
	public static IntersectionResult boxSphereIntersections(double[] z, double[] d, double[] lb, double[] ub, double trustRadius, boolean entireLine) {
		return boxSphereIntersectionsWithExtraInfo(z, d, lb, ub, trustRadius, entireLine)[0];
	}

	/**
	 * Convenience overload of
	 * {@link #boxSphereIntersectionsWithExtraInfo(double[], double[], double[], double[], double, boolean)}
	 * with {@code entireLine = false}.
	 *
	 * @param z           initial point
	 * @param d           direction
	 * @param lb          box lower bounds
	 * @param ub          box upper bounds
	 * @param trustRadius sphere radius
	 * @return three-element array {@code {boxSphere, sphere, box}} of
	 *         {@link IntersectionResult}
	 */
	public static IntersectionResult[] boxSphereIntersectionsWithExtraInfo(double[] z, double[] d, double[] lb, double[] ub, double trustRadius) {
		boolean entireLine = false;
		return boxSphereIntersectionsWithExtraInfo(z, d, lb, ub, trustRadius, entireLine);
	}

	/**
	 * Find the intersection between segment (or line) and box/sphere constraints.
	 * <p>
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
	 * Return clipped value of x.
	 *
	 * @param x  [n] position vector to force into bounds
	 * @param lb [n] lower bounds
	 * @param ub [n] upper bounds
	 * @return coerced copy of {@code x} such that {@code lb &le; x &le; ub} for all entries
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
	 * Check whether {@code lb &le; x &le; ub} elementwise.
	 *
	 * @param x  length-{@code n} position to test
	 * @param lb length-{@code n} lower bounds
	 * @param ub length-{@code n} upper bounds
	 * @return {@code true} iff every entry satisfies {@code lb[i] &le; x[i] &le; ub[i]}
	 */
	public static boolean insideBoxBoundaries(double[] x, double[] lb, double[] ub) {
		for (int i=0; i<x.length; ++i) {
			if (x[i] < lb[i] || x[i] > ub[i]) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Approximately minimize {@code 1/2 ||A x + b||^2} subject to
	 * {@code ||x|| &lt; Delta} and {@code lb &le; x &le; ub} using a modified dogleg
	 * approach. Implementation based on Byrd-Hribar-Nocedal (1999),
	 * pp.885-886.
	 *
	 * @param A {@code m x n} matrix in the minimization problem
	 *          ({@code m &lt; n} expected)
	 * @param Y {@link Matrix} or {@link LinearOperator} that applies the
	 *          projection {@code Q = A^T (A A^T)^-^1}; for any input {@code x},
	 *          {@code y = Q x} is the minimum-norm solution of {@code A y = x}
	 * @param b length-{@code m} vector in the minimization problem
	 * @param trustRadius trust radius {@code Delta} (sphere boundary)
	 * @param lb length-{@code n} lower bounds; {@code Double.NEGATIVE_INFINITY}
	 *           disables the bound for that component. Algorithm assumes
	 *           {@code lb &le; 0}
	 * @param ub length-{@code n} upper bounds; {@code Double.POSITIVE_INFINITY}
	 *           disables the bound for that component. Algorithm assumes
	 *           {@code ub &ge; 0}
	 * @return length-{@code n} solution vector
	 */
	public static Matrix modifiedDogleg(Matrix A, Object Y, Matrix b, double trustRadius, double[] lb, double[] ub) {

		// Compute minimum norm minimizer of 1/2*|| A x + b ||^2.
		Matrix newtonPoint;
		if (Y instanceof Matrix) {
			newtonPoint = ((Matrix) Y).mtimes(b).times(-1);
		} else if (Y instanceof LinearOperator) {
			newtonPoint = ((LinearOperator) Y).apply(b).times(-1);
		} else {
			throw new RuntimeException("Y must be of type Matrix or LinearOperator");
		}

		if (insideBoxBoundaries(newtonPoint.toColumnArray(), lb, ub)
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
		IntersectionResult r1 = boxSphereIntersections(z.toColumnArray(), p.toColumnArray(), lb, ub, trustRadius);
		double alpha = r1.tB();

		if (!r1.intersect()) {
			// Check the segment between the origin and cauchy_point for a possible solution.
			z = origin;
			p = cauchyPoint;
			IntersectionResult r2 = boxSphereIntersections(z.toColumnArray(), p.toColumnArray(), lb, ub, trustRadius);
			alpha = r2.tB();
		}
		Matrix x1 = z.plus(p.times(alpha));

		// Check the segment between origin and newton_point for a possible solution.
		z = origin;
		p = newtonPoint;
		IntersectionResult r3 = boxSphereIntersections(z.toColumnArray(), p.toColumnArray(), lb, ub, trustRadius);
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
	 * Solve the equality-constrained quadratic program
	 * <pre>
	 *   minimize    1/2 x^T H x + x^T c
	 *   subject to  A x + b = 0
	 * </pre>
	 * with the projected conjugate-gradient method. Convenience overload
	 * with no trust-region or box constraints.
	 *
	 * @param H operator that computes {@code H v} on a length-{@code n} vector
	 * @param c length-{@code n} gradient of the quadratic objective
	 * @param Z {@link Matrix} or {@link LinearOperator} that projects
	 *          {@code x} into the null space of {@code A}
	 * @param Y {@link Matrix} or {@link LinearOperator} that, given
	 *          {@code b}, returns the minimum-norm solution of {@code A x + b = 0}
	 * @param b length-{@code m} right-hand side of the constraint equation
	 * @return solution vector and additional CG info
	 */
	public static CGInfo projectedCG(LinearOperator H, Matrix c, Object Z, Object Y, Matrix b) {

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = null;
		double[] ub = null;
		double tolerance = Double.NaN;

		return projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tolerance);
	}

	/**
	 * Solve EQP problem with projected CG method.
	 * <p>
	 * Solve equality-constrained quadratic programming problem
	 * {@code min 1/2 x^T H x + x^t c} subject to {@code A x + b = 0} and, possibly,
	 * to trust region constraints {@code ||x|| < trust_radius} and box constraints
	 * {@code lb <= x <= ub}.
	 *
	 * <p>Without spherical or box constraints and with enough iterations, the
	 * method returns the exact optimum. With those constraints active, the
	 * returned value is only an inexpensive approximation.
	 *
	 * @param H operator that computes {@code H v} on a length-{@code n} vector
	 * @param c length-{@code n} gradient of the quadratic objective
	 * @param Z {@link Matrix} or {@link LinearOperator} that projects {@code x}
	 *          into the null space of {@code A}
	 * @param Y {@link Matrix} or {@link LinearOperator} that, given {@code b},
	 *          returns the minimum-norm solution of {@code A x + b = 0}
	 * @param b length-{@code m} right-hand side of the constraint equation
	 * @param trustRadius trust-region radius; {@code Double.POSITIVE_INFINITY}
	 *                    disables the trust-region constraint
	 * @param lb length-{@code n} lower bounds; {@code Double.NEGATIVE_INFINITY}
	 *           disables the bound for that component
	 * @param ub length-{@code n} upper bounds; {@code Double.POSITIVE_INFINITY}
	 *           disables the bound for that component
	 * @param tolerance termination tolerance for the projected residual
	 * @return solution vector and additional CG info
	 */
	public static CGInfo projectedCG(LinearOperator H, Matrix c, Object Z, Object Y, Matrix b, double trustRadius, double[] lb,
			double[] ub, double tolerance) {

		long maxIterations = -1;
		long maxInfeasibleIterations = -1;
		boolean returnAll = false;

		return projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tolerance, maxIterations, maxInfeasibleIterations, returnAll);
	}

	/**
	 * Solve the equality-constrained quadratic program
	 * <pre>
	 *   minimize    1/2 x^T H x + x^T c
	 *   subject to  A x + b = 0
	 *   and (optionally) ||x|| &lt; trustRadius, lb &le; x &le; ub
	 * </pre>
	 * with the projected conjugate-gradient method (Algorithm 6.2 of
	 * Gould-Hribar-Nocedal, 2001).
	 *
	 * <p>Without spherical or box constraints and with enough iterations, the
	 * method returns the exact optimum. With those constraints active, the
	 * returned value is only an inexpensive approximation.
	 *
	 * @param H operator that computes {@code H v}
	 * @param c length-{@code n} gradient of the quadratic objective
	 * @param Z {@link Matrix} or {@link LinearOperator} projecting into null({@code A})
	 * @param Y {@link Matrix} or {@link LinearOperator} for the minimum-norm
	 *          solution of {@code A x + b = 0}
	 * @param b length-{@code m} right-hand side of the constraint equation
	 * @param trustRadius trust-region radius; {@code Double.POSITIVE_INFINITY}
	 *                    disables the trust-region constraint
	 * @param lb length-{@code n} lower bounds (use {@code -inf} to disable per dim)
	 * @param ub length-{@code n} upper bounds (use {@code +inf} to disable per dim)
	 * @param tolerance termination tolerance for the projected residual
	 * @param maxIterations max CG iterations; {@code -1} for the default
	 *                      {@code n - m}
	 * @param maxInfeasibleIterations max iterations spent infeasible w.r.t. box
	 *                                constraints; {@code -1} for the default
	 *                                {@code n - m}
	 * @param returnAll if {@code true}, the result includes the iterate trajectory
	 * @return solution vector and additional CG info
	 */
	public static CGInfo projectedCG(LinearOperator H, Matrix c, Object Z, Object Y, Matrix b, double trustRadius, double[] lb,
			double[] ub, double tolerance, long maxIterations, long maxInfeasibleIterations, boolean returnAll) {

		CGInfo result = new CGInfo();
		if (returnAll) {
			result.allVecs = new LinkedList<>();
		}

		final double CLOSE_TO_ZERO = 1.0e-25;

		final long n = c.getRowCount(); // Number of parameters
		final long m = b.getRowCount(); // Number of constraints

		// Initial Values
		if (Y instanceof Matrix) {
			result.x = ((Matrix) Y).mtimes(b.times(-1));
		} else if (Y instanceof LinearOperator) {
			result.x = ((LinearOperator) Y).apply(b.times(-1));
		} else {
			throw new RuntimeException("Y has to be Matrix or LinearOperator");
		}

		Matrix r, g;
		if (Z instanceof Matrix) {
			r = ((Matrix) Z).mtimes(H.apply(result.x).plus(c));
			g = ((Matrix) Z).mtimes(r);
		} else if (Y instanceof LinearOperator) {
			r = ((LinearOperator) Z).apply(H.apply(result.x).plus(c));
			g = ((LinearOperator) Z).apply(r);
		} else {
			throw new RuntimeException("Z has to be Matrix or LinearOperator");
		}
		Matrix p = g.times(-1);

		// Store {@code x} value
		if (returnAll) {
			result.allVecs.add(Matrix.Factory.copyFromMatrix(result.x));
		}

		// If x > trust-region the problem does not have a solution.
		double trDistance = trustRadius - result.x.norm2();
		if (trDistance < 0.0) {
			throw new RuntimeException("Trust region problem does not have a solution.");
		} else if (trDistance < CLOSE_TO_ZERO) {
			// If x == trust_radius, then x is the solution
			// to the optimization problem, since x is the
			// minimum norm solution to Ax=b.

			result.niter = 0;
			result.stopCond = PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED;
			result.hitsBoundary = true;
			if (returnAll) {
				result.allVecs.add(Matrix.Factory.copyFromMatrix(result.x));
			}

			return result;
		}

		// Values for the first iteration
		Matrix H_p = H.apply(p);
		double gNorm = g.norm2();
		double rt_g = gNorm*gNorm; // g.T g = r.T Z g = r.T g (ref [1], p.1389)

		// Set default tolerance
		if (Double.isNaN(tolerance)) {
			tolerance = Math.max(Math.min(0.01*Math.sqrt(rt_g), 0.1*rt_g), CLOSE_TO_ZERO);
		}

		// Set default lower and upper bounds
		if (lb == null) {
			lb = new double[(int) n];
			Arrays.fill(lb, Double.NEGATIVE_INFINITY);
		}
		if (ub == null) {
			ub = new double[(int) n];
			Arrays.fill(ub, Double.POSITIVE_INFINITY);
		}

		// Set maximum iterations
		if (maxIterations == -1) {
			maxIterations = n-m; // default value
		}
		maxIterations = Math.min(maxIterations, n-m); // limit to max. n-m

		// Set maximum infeasible iterations
		if (maxInfeasibleIterations == -1) {
			maxInfeasibleIterations = n-m; // default value
		}

		result.hitsBoundary = false;
		result.stopCond = PCGStoppingCondition.ITER_LIMIT_REACHED;
		int counter = 0;
		Matrix lastFeasibleX = Matrix.Factory.zeros(result.x.getRowCount(), result.x.getColumnCount());
		int k = 0;
		for (int i=0; i<maxIterations; ++i) {
			// Stop criteria - Tolerance : r.T g < tol
			if (rt_g < tolerance) {
				result.stopCond = PCGStoppingCondition.TOLERANCE_SATISFIED;
				break;
			}

			k++;

			// Compute curvature
			double pt_H_g = H_p.transpose().mtimes(p).doubleValue();

			// Stop criteria - Negative curvature
			if (pt_H_g <= 0.0) {
				if (Double.isInfinite(trustRadius)) {
					throw new RuntimeException("Negative curvature not allowed for unrestricted problems.");
				} else {
					// Find intersection with constraints
					IntersectionResult ir = boxSphereIntersections(result.x.toColumnArray(), p.toColumnArray(), lb, ub, trustRadius, true);
					double alpha = ir.tB();

					// Update solution
					if (ir.intersect()) {
						result.x = result.x.plus(p.times(alpha));
					}

					// Reinforce variables are inside box constraints.
	                // This is only necessary because of roundoff errors.
					result.x = Matrix.Factory.linkToArray(reinforceBoxBoundaries(result.x.toColumnArray(), lb, ub));

					// Attribute information
					result.stopCond = PCGStoppingCondition.NEGATIVE_CURVATURE;
					result.hitsBoundary = true;
					break;
				}
			}

			// Get next step
			double alpha = rt_g / pt_H_g;
			Matrix xNext = result.x.plus(p.times(alpha));

			// Stop criteria - Hits boundary
			if (xNext.norm2() >= trustRadius) {
				// Find intersection with box constraints
				IntersectionResult ir = boxSphereIntersections(result.x.toColumnArray(), p.times(alpha).toColumnArray(), lb, ub, trustRadius);
				double theta = ir.tB();

				// Update solution
				if (ir.intersect()) {
					result.x = result.x.plus(p.times(alpha*theta));
				}

				// Reinforce variables are inside box constraints.
                // This is only necessary because of roundoff errors.
				result.x = Matrix.Factory.linkToArray(reinforceBoxBoundaries(result.x.toColumnArray(), lb, ub));

				// Attribute information
				result.stopCond = PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED;
				result.hitsBoundary = true;
				break;
			}

			// Check if {@code x} is inside the box and start counter if it is not.
			if (insideBoxBoundaries(xNext.toColumnArray(), lb, ub)) {
				counter = 0;
			} else {
				counter++;
			}

			// Whenever outside box constraints keep looking for intersections.
			if (counter > 0) {
				IntersectionResult ir = boxSphereIntersections(result.x.toColumnArray(), p.times(alpha).toColumnArray(), lb, ub, trustRadius);
				double theta = ir.tB();

				if (ir.intersect()) {
					lastFeasibleX = result.x.plus(p.times(alpha*theta));

					// Reinforce variables are inside box constraints.
	                // This is only necessary because of roundoff errors.
					lastFeasibleX = Matrix.Factory.linkToArray(reinforceBoxBoundaries(lastFeasibleX.toColumnArray(), lb, ub));

					counter = 0;
				}
			}

			// Stop after too many infeasible (regarding box constraints) iteration.
			if (counter > maxInfeasibleIterations) {
				break;
			}

			// Store ``x_next`` value
			if (returnAll) {
				result.allVecs.add(Matrix.Factory.copyFromMatrix(xNext));
			}

			// Update residual
			Matrix rNext = r.plus(H_p.times(alpha));

			// Project residual g+ = Z r+
			Matrix gNext;
			if (Z instanceof Matrix) {
				gNext = ((Matrix) Z).mtimes(rNext);
			} else if (Z instanceof LinearOperator) {
				gNext = ((LinearOperator) Z).apply(rNext);
			} else {
				throw new RuntimeException("Z must be either Matrix or LinearOperator");
			}

			// Compute conjugate direction step d
			double normGNext = gNext.norm2();
			double rt_g_next = normGNext*normGNext; // g.T g = r.T g (ref [1] p.1389)

			double beta = rt_g_next / rt_g;
			p = gNext.times(-1).plus(p.times(beta));

			// Prepare for next iteration

			result.x = xNext;
			g = gNext;
			r = gNext;
			double normG = g.norm2();
			rt_g = normG*normG; // g.T g = r.T Z g = r.T g (ref [1] p.1389)
			H_p = H.apply(p);
		}

		if (!insideBoxBoundaries(result.x.toColumnArray(), lb, ub)) {
			result.x = lastFeasibleX;
			result.hitsBoundary = true;
		}

		result.niter = k;

		return result;
	}
}
