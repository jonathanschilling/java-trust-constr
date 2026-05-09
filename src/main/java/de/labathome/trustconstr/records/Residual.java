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

/**
 * Lower and upper residuals for a two-sided constraint
 * {@code lb &le; f(x) &le; ub}. Returned by
 * {@link de.labathome.trustconstr.LinearConstraint#residual(de.labathome.trustconstr.matrix.Matrix)}
 * and {@link Bounds#residual(de.labathome.trustconstr.matrix.Matrix)}.
 *
 * <p>Mirrors scipy's tuple return shape from
 * {@code LinearConstraint.residual(x)} and {@code Bounds.residual(x)}:
 *
 * <pre>
 *     sl = f(x) - lb     (non-negative iff lower bound is satisfied)
 *     sb = ub - f(x)     (non-negative iff upper bound is satisfied)
 * </pre>
 *
 * <p>Both components are non-negative iff every row of the constraint is
 * satisfied. A negative entry indicates the corresponding row is violated.
 *
 * @param sl lower residual ({@code f(x) - lb})
 * @param sb upper residual ({@code ub - f(x)})
 */
public record Residual(double[] sl, double[] sb) {}
