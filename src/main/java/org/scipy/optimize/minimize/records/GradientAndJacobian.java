package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public record GradientAndJacobian(Matrix grad, Matrix jac) {
}
