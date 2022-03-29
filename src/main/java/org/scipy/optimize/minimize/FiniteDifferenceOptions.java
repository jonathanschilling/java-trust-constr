package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;

public record FiniteDifferenceOptions(FiniteDifferenceMethod method,
		double relStep, double[] absStep,
		FiniteDifferenceBounds bounds, boolean asLinearOperator){
}
