package org.scipy.optimize.minimize.records;

public record Bounds (double[] lb, double[] ub, boolean keepFeasible) {

}
