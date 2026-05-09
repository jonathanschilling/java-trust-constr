/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
