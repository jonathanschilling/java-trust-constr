package de.labathome.trust_constr;

/**
 * Trust-region interior point method.
 *
 * @see [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
 *          "An interior point algorithm for large-scale nonlinear
 *          programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
 * @see [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
 *          "On the local behavior of an interior point method for
 *          nonlinear programming." Numerical analysis 1997 (1997): 37-56.
 * @see [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
 *          Second Edition (2006).
 */
public class TrustRegionInteriorPoint {

	/**
	 * Trust-region interior points method.
	 *
	 * Solve problem:
	 * <pre>
	 * minimize fun(x)
	 * subject to: constr_ineq(x) <= 0
	 *             constr_eq(x) = 0
	 * </pre>
	 * using trust-region interior point method described in [1].
	 */
	public static void trustRegionInteriorPoint() {






	}



}
