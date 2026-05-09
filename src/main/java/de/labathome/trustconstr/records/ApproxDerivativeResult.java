package de.labathome.trustconstr.records;

import java.util.Map;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Output shape of {@link de.labathome.trustconstr.NumDiff#approxDerivativeFullOutput}.
 * Mirrors scipy's {@code approx_derivative(..., full_output=True)} 2-tuple
 * {@code (jac, info_dict)}.
 *
 * @param jacobian computed FD Jacobian ({@code m x n})
 * @param info     metadata about the FD calculation; keys include
 *                 {@code "nfev"} ({@link Integer}: number of function
 *                 evaluations consumed) and {@code "method"}
 *                 ({@link de.labathome.trustconstr.enums.FiniteDifferenceMethod})
 */
public record ApproxDerivativeResult(Matrix jacobian, Map<String, Object> info) {}
