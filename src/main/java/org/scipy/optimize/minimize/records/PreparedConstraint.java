package org.scipy.optimize.minimize.records;

import java.util.Optional;

import org.scipy.optimize.minimize.VectorFunction;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.ujmp.core.Matrix;

public class PreparedConstraint {

	private VectorFunction fun;
	private Bounds bounds;

	public PreparedConstraint(Constraint c, Matrix x0, Optional<Boolean> sparseJacobian, FiniteDifferenceBounds finiteDiffBounds) {


	}

	public PreparedConstraint(Bounds bounds, Matrix x0, Optional<Boolean> sparseJacobian) {

	}

	public Bounds bounds() {
		return bounds;
	}

	public VectorFunction fun() {
		return fun;
	}
}

