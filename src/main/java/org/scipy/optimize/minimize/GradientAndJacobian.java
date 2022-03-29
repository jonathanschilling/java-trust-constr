package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public record GradientAndJacobian(Matrix grad, Matrix jac) {
}
