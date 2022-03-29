package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.State;
import org.ujmp.core.Matrix;

public record StatefulResult(
	Matrix x,
	State state) {
}
