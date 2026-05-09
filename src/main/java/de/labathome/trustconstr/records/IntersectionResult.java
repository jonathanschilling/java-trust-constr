package de.labathome.trustconstr.records;

/**
 * Result of intersecting a parametric segment {@code x(t) = z + t*d} with a
 * trust-region or box constraint. The segment lies inside the constraint set
 * for {@code tA &le; t &le; tB} when {@link #intersect()} is {@code true}.
 */
public final class IntersectionResult {

	private double tA;
	private double tB;
	private boolean intersect;

	/**
	 * @param tA        lower end of the {@code t}-interval
	 * @param tB        upper end of the {@code t}-interval
	 * @param intersect {@code true} iff the segment enters the feasible region
	 */
	public IntersectionResult(double tA, double tB, boolean intersect) {
		this.tA = tA;
		this.tB = tB;
		this.intersect = intersect;
	}

	/** @return lower end of the {@code t}-interval */
	public double tA() {
		return tA;
	}

	/** @return upper end of the {@code t}-interval */
	public double tB() {
		return tB;
	}

	/** @return whether the segment enters the feasible region at all */
	public boolean intersect() {
		return intersect;
	}
}
