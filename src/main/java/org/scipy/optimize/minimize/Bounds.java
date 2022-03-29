package org.scipy.optimize.minimize;

public record Bounds (double[] lb, double[] ub, boolean keepFeasible) {

}
