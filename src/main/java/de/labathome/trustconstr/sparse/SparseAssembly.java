package de.labathome.trustconstr.sparse;

/**
 * Sparse-aware structured assembly: vstack, hstack, blockArray, and the
 * specialised KKT/Jacobian builders that {@code trust-constr} relies on.
 *
 * <p>These mirror the slice of {@code scipy.sparse} that the trust-region
 * constrained optimiser actually uses. The block constructors in scipy live in
 * {@code scipy.sparse._construct} and are reached via {@code sps.vstack},
 * {@code sps.hstack}, {@code sps.block_array}.
 */
public final class SparseAssembly {

	private SparseAssembly() { }

	/**
	 * Vertical stack of CSR matrices: rows are concatenated, columns must agree.
	 *
	 * Empty input yields an empty {@code 0 x 0} matrix; if all blocks have a
	 * common positive column count, the result has that column count.
	 *
	 * @param blocks CSR blocks to stack (in row order)
	 * @return concatenated CSR matrix
	 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html">scipy.sparse.vstack</a>
	 */
	public static CSRMatrix vstack(CSRMatrix... blocks) {
		if (blocks.length == 0) {
			return new CSRMatrix(0, 0, new int[]{0}, new int[0], new double[0]);
		}
		int cols = blocks[0].cols();
		int totalRows = 0;
		int totalNnz = 0;
		for (CSRMatrix b : blocks) {
			if (b.cols() != cols) {
				throw new IllegalArgumentException("vstack: column count mismatch");
			}
			totalRows += b.rows();
			totalNnz += b.nnz();
		}
		int[] indptr = new int[totalRows + 1];
		int[] indices = new int[totalNnz];
		double[] data = new double[totalNnz];
		int rowOffset = 0;
		int dataOffset = 0;
		for (CSRMatrix b : blocks) {
			int[] bp = b.indptrRef();
			int[] bi = b.indicesRef();
			double[] bd = b.dataRef();
			for (int i = 0; i < b.rows(); ++i) {
				indptr[rowOffset + i] = dataOffset + bp[i];
			}
			System.arraycopy(bi, 0, indices, dataOffset, b.nnz());
			System.arraycopy(bd, 0, data, dataOffset, b.nnz());
			rowOffset += b.rows();
			dataOffset += b.nnz();
		}
		indptr[totalRows] = totalNnz;
		return new CSRMatrix(totalRows, cols, indptr, indices, data);
	}

	/**
	 * Horizontal stack of CSR matrices: columns are concatenated, rows must agree.
	 *
	 * @param blocks CSR blocks to stack (in column order)
	 * @return concatenated CSR matrix
	 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html">scipy.sparse.hstack</a>
	 */
	public static CSRMatrix hstack(CSRMatrix... blocks) {
		if (blocks.length == 0) {
			return new CSRMatrix(0, 0, new int[]{0}, new int[0], new double[0]);
		}
		int rows = blocks[0].rows();
		int totalCols = 0;
		int totalNnz = 0;
		int[] colOffsets = new int[blocks.length];
		for (int b = 0; b < blocks.length; ++b) {
			if (blocks[b].rows() != rows) {
				throw new IllegalArgumentException("hstack: row count mismatch");
			}
			colOffsets[b] = totalCols;
			totalCols += blocks[b].cols();
			totalNnz += blocks[b].nnz();
		}
		int[] indptr = new int[rows + 1];
		int[] indices = new int[totalNnz];
		double[] data = new double[totalNnz];
		int k = 0;
		for (int i = 0; i < rows; ++i) {
			indptr[i] = k;
			for (int b = 0; b < blocks.length; ++b) {
				int[] bp = blocks[b].indptrRef();
				int[] bi = blocks[b].indicesRef();
				double[] bd = blocks[b].dataRef();
				int colOffset = colOffsets[b];
				for (int kb = bp[i]; kb < bp[i + 1]; ++kb) {
					indices[k] = bi[kb] + colOffset;
					data[k] = bd[kb];
					++k;
				}
			}
		}
		indptr[rows] = totalNnz;
		return new CSRMatrix(rows, totalCols, indptr, indices, data);
	}

	/**
	 * 2D block assembly. {@code blocks[i][j]} is the block at row-band i,
	 * column-band j; {@code null} represents a zero block. Within a row band all
	 * non-null blocks must share row count; within a column band all non-null
	 * blocks must share column count. At least one block per row band and per
	 * column band must be non-null so the band size can be inferred.
	 *
	 * @param blocks 2-D grid of {@link CSRMatrix} blocks (rows by columns); {@code null} = zero block
	 * @return assembled CSR matrix
	 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_array.html">scipy.sparse.block_array</a>
	 */
	public static CSRMatrix blockArray(CSRMatrix[][] blocks) {
		int br = blocks.length;
		if (br == 0) {
			return new CSRMatrix(0, 0, new int[]{0}, new int[0], new double[0]);
		}
		int bc = blocks[0].length;
		for (int i = 0; i < br; ++i) {
			if (blocks[i].length != bc) {
				throw new IllegalArgumentException("blockArray: ragged block grid");
			}
		}
		int[] rowSizes = new int[br];
		int[] colSizes = new int[bc];
		for (int i = 0; i < br; ++i) rowSizes[i] = -1;
		for (int j = 0; j < bc; ++j) colSizes[j] = -1;
		for (int i = 0; i < br; ++i) {
			for (int j = 0; j < bc; ++j) {
				CSRMatrix b = blocks[i][j];
				if (b == null) continue;
				if (rowSizes[i] == -1) rowSizes[i] = b.rows();
				else if (rowSizes[i] != b.rows()) {
					throw new IllegalArgumentException("blockArray: row-band " + i + " has inconsistent row counts");
				}
				if (colSizes[j] == -1) colSizes[j] = b.cols();
				else if (colSizes[j] != b.cols()) {
					throw new IllegalArgumentException("blockArray: column-band " + j + " has inconsistent column counts");
				}
			}
		}
		for (int i = 0; i < br; ++i) {
			if (rowSizes[i] == -1) {
				throw new IllegalArgumentException("blockArray: row-band " + i + " is entirely null");
			}
		}
		for (int j = 0; j < bc; ++j) {
			if (colSizes[j] == -1) {
				throw new IllegalArgumentException("blockArray: column-band " + j + " is entirely null");
			}
		}
		// Build each band-row by hstacking, then vstack the bands.
		CSRMatrix[] bandRows = new CSRMatrix[br];
		for (int i = 0; i < br; ++i) {
			CSRMatrix[] rowBlocks = new CSRMatrix[bc];
			for (int j = 0; j < bc; ++j) {
				rowBlocks[j] = blocks[i][j] == null
						? zeros(rowSizes[i], colSizes[j])
						: blocks[i][j];
			}
			bandRows[i] = hstack(rowBlocks);
		}
		return vstack(bandRows);
	}

	/**
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return a {@code rows x cols} CSR matrix with no stored entries
	 */
	public static CSRMatrix zeros(int rows, int cols) {
		int[] indptr = new int[rows + 1];
		return new CSRMatrix(rows, cols, indptr, new int[0], new double[0]);
	}

	/**
	 * Build the augmented Jacobian used inside the interior-point method:
	 *
	 * <pre>
	 *     [ J_eq      0     ]
	 *     [ J_ineq   diag(s) ]
	 * </pre>
	 *
	 * Equivalent to {@code blockArray([[J_eq, null], [J_ineq, diag(s)]])} but
	 * inlined for the specific structure -- this is the optimised path
	 * {@code _assemble_sparse_jacobian} from {@code tr_interior_point.py:187}.
	 *
	 * @param jEq   {@code n_eq x n_vars} CSR equality Jacobian
	 * @param jIneq {@code n_ineq x n_vars} CSR inequality Jacobian
	 * @param s     {@code n_ineq} slack-variable values
	 * @return {@code (n_eq + n_ineq) x (n_vars + n_ineq)} CSR
	 */
	public static CSRMatrix assembleJacobianWithSlacks(CSRMatrix jEq, CSRMatrix jIneq, double[] s) {
		int nEq = jEq.rows();
		int nIneq = jIneq.rows();
		int nVars = jEq.cols();
		if (jIneq.cols() != nVars) {
			throw new IllegalArgumentException("J_eq and J_ineq must have the same column count");
		}
		if (s.length != nIneq) {
			throw new IllegalArgumentException("s length must equal n_ineq");
		}
		// Step 1: vstack J_eq and J_ineq to get J_aux.
		CSRMatrix jAux = vstack(jEq, jIneq);
		int[] auxIndptr = jAux.indptrRef();
		int[] auxIndices = jAux.indicesRef();
		double[] auxData = jAux.dataRef();
		// Step 2: shift indptr -- equality rows unchanged, inequality row i (i=0..n_ineq-1)
		// gets one extra entry shifted in: row offset increases by (i+1).
		int totalRows = nEq + nIneq;
		int[] newIndptr = new int[totalRows + 1];
		for (int i = 0; i <= nEq; ++i) {
			newIndptr[i] = auxIndptr[i];
		}
		for (int i = 0; i < nIneq; ++i) {
			newIndptr[nEq + i + 1] = auxIndptr[nEq + i + 1] + (i + 1);
		}
		int newNnz = auxIndices.length + nIneq;
		int[] newIndices = new int[newNnz];
		double[] newData = new double[newNnz];
		// Copy equality rows verbatim into the front.
		int auxNnzEq = auxIndptr[nEq];
		System.arraycopy(auxIndices, 0, newIndices, 0, auxNnzEq);
		System.arraycopy(auxData, 0, newData, 0, auxNnzEq);
		// For each inequality row, copy the row data and append the slack entry.
		int writePos = auxNnzEq;
		for (int i = 0; i < nIneq; ++i) {
			int rowStart = auxIndptr[nEq + i];
			int rowEnd = auxIndptr[nEq + i + 1];
			int rowLen = rowEnd - rowStart;
			System.arraycopy(auxIndices, rowStart, newIndices, writePos, rowLen);
			System.arraycopy(auxData, rowStart, newData, writePos, rowLen);
			writePos += rowLen;
			// Append the slack column entry: column = n_vars + i, value = s[i].
			newIndices[writePos] = nVars + i;
			newData[writePos] = s[i];
			++writePos;
		}
		return new CSRMatrix(totalRows, nVars + nIneq, newIndptr, newIndices, newData);
	}
}
