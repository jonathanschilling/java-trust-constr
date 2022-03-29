package org.scipy.optimize.minimize;

/**
 * The line/segment {@code x(t) = z + t*d} is inside the ball for
 * {@code tA <= t <= tB}.
 */
public record IntersectionResult(double tA, double tB, boolean intersect) {

}
