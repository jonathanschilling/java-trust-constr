package org.scipy.optimize.minimize;

import java.util.Optional;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.ujmp.core.Matrix;

public class PreparedConstraint {

	public VectorFunction fun;
	public Bounds bounds;

	public PreparedConstraint(Constraint c, Matrix x0, Optional<Boolean> sparseJacobian, FiniteDifferenceBounds finiteDiffBounds) {


	}

	public PreparedConstraint(Bounds bounds, Matrix x0, Optional<Boolean> sparseJacobian) {

	}
}

