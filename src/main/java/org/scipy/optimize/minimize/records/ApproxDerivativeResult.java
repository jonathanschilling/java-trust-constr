package org.scipy.optimize.minimize.records;

import java.util.Map;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Output shape of {@link org.scipy.optimize.minimize.NumDiff#approxDerivativeFullOutput}.
 * Mirrors scipy's {@code approx_derivative(..., full_output=True)} 2-tuple
 * {@code (jac, info_dict)}.
 *
 * @param jacobian computed FD Jacobian ({@code m x n})
 * @param info     metadata about the FD calculation; keys include
 *                 {@code "nfev"} ({@link Integer}: number of function
 *                 evaluations consumed) and {@code "method"}
 *                 ({@link org.scipy.optimize.minimize.enums.FiniteDifferenceMethod})
 */
public record ApproxDerivativeResult(Matrix jacobian, Map<String, Object> info) {}
