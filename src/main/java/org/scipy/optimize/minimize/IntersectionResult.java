package org.scipy.optimize.minimize;

/**
 * The line/segment {@code x(t) = z + t*d} is inside the ball for
 * {@code tA <= t <= tB}.
 */
public class IntersectionResult {

	/** start value for line parameter */
	protected double tA;

	/** end value for line parameter */
	protected double tB;

	/**
	 * When {@code true}, there is a intersection between the line/segment and the
	 * sphere. On the other hand, when {@code false}, there is no intersection.
	 */
	protected boolean intersect;

	/**
	 * The line/segment {@code x(t) = z + t*d} is inside the ball for
	 * {@code tA <= t <= tB}.
	 *
	 * @param tA        start value for line parameter
	 * @param tB        end value for line parameter
	 * @param intersect When {@code true}, there is a intersection between the
	 *                  line/segment and the sphere. On the other hand, when
	 *                  {@code false}, there is no intersection.
	 */
	public IntersectionResult(double tA, double tB, boolean intersect) {
		this.tA = tA;
		this.tB = tB;
		this.intersect = intersect;
	}

	/**
	 * Get the start value for line parameter.
	 *
	 * @return start value for line parameter
	 */
	public double tA() {
		return tA;
	}

	/**
	 * Get the end value for line parameter.
	 *
	 * @return end value for line parameter
	 */
	public double tB() {
		return tB;
	}

	/**
	 * When {@code true}, there is a intersection between the line/segment and the
	 * sphere. On the other hand, when {@code false}, there is no intersection.
	 *
	 * @return true if an intersection between line and ball occured
	 */
	public boolean intersect() {
		return intersect;
	}
}
