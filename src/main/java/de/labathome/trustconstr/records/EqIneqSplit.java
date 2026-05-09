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

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Pair of canonical equality and inequality components produced by a
 * {@link de.labathome.trustconstr.CanonicalConstraint}'s {@code fun(x)}
 * or {@code jac(x)}. Mirrors scipy's tuple return shape from
 * {@code canonical_constraint.py}.
 *
 * <p>For {@code fun(x)}: {@code eq} is {@code nEq x 1}, {@code ineq} is
 * {@code nIneq x 1} (column vectors of canonical residuals).
 *
 * <p>For {@code jac(x)}: {@code eq} is {@code nEq x n}, {@code ineq} is
 * {@code nIneq x n} (canonical Jacobian blocks).
 *
 * @param eq   equality component
 * @param ineq inequality component
 */
public record EqIneqSplit(Matrix eq, Matrix ineq) {}
