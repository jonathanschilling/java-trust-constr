package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public record FunctionAndConstraint(double f, Matrix c) {
}
