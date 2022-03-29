package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public record FunctionAndConstraint(double f, Matrix c) {
}
