package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public record StatefulResult(
	Matrix x,
	State state) {
}
