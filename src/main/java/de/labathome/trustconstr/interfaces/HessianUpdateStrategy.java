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
package de.labathome.trustconstr.interfaces;

import java.util.function.BiFunction;

import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Interface for implementing Hessian update strategies.
 *
 * Many optimization methods make use of Hessian (or inverse Hessian)
 * approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
 * Some of these  approximations, however, do not actually need to store
 * the entire matrix or can compute the internal matrix product with a
 * given vector in a very efficiently manner. This class serves as an
 * abstract interface between the optimization algorithm and the
 * quasi-Newton update strategies, giving freedom of implementation
 * to store and update the internal matrix as efficiently as possible.
 * Different choices of initialization and update procedure will result
 * in different quasi-Newton strategies.
 *
 * Four methods should be implemented in derived classes: ``initialize``,
 * ``update``, ``dot`` and ``get_matrix``.
 *
 * Any instance of a class that implements this interface,
 * can be accepted by the method ``minimize`` and used by
 * the compatible solvers to approximate the Hessian (or
 * inverse Hessian) used by the optimization algorithms.
 */
public interface HessianUpdateStrategy extends BiFunction<Matrix, Object, Matrix> {

	/**
	 * Initialize internal matrix.
	 * <p>
	 * Allocate internal memory for storing and updating
	 * the Hessian or its inverse.
	 *
	 * @param n Problem dimension.
	 * @param approxType Selects either the Hessian or the inverse Hessian.
     *                   When set to 'hess' the Hessian will be stored and updated.
     *                   When set to 'inv_hess' its inverse will be used instead.
	 */
	public void initialize(long n, HessianApproximationType approxType);

	/**
	 * Update internal matrix.
	 * <p>
	 * Update Hessian matrix or its inverse (depending on how 'approx_type'
	 * is defined) using information about the last evaluated points.
	 *
	 * @param deltaX The difference between two points the gradient
	 *               function have been evaluated at: ``delta_x = x2 - x1``.
	 * @param deltaG The difference between the gradients:
            		 ``delta_grad = grad(x2) - grad(x1)``.
	 */
	public void update(Matrix deltaX, Matrix deltaG);

	/**
	 * Return current internal matrix.
	 * @return Dense matrix containing either the Hessian or its inverse
	 *         (depending on how 'approx_type' is defined).
	 */
	public Matrix getMatrix();

}
