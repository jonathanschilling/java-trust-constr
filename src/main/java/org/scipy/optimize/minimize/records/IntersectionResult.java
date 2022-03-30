package org.scipy.optimize.minimize.records;

/**
 * The line/segment {@code x(t) = z + t*d} is inside the ball for
 * {@code tA <= t <= tB}.
 */
public final class IntersectionResult {

	private double tA;
	private double tB;
	private boolean intersect;

	public IntersectionResult(double tA, double tB, boolean intersect) {
		this.tA = tA;
		this.tB = tB;
		this.intersect = intersect;
	}

	public double tA() {
		return tA;
	}

	public double tB() {
		return tB;
	}

	public boolean intersect() {
		return intersect;
	}
}
