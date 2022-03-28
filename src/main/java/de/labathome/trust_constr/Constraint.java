package de.labathome.trust_constr;
import org.ujmp.core.Matrix;
public interface Constraint {
	public Matrix constrEq(Matrix x);
	public Matrix constrIneq(Matrix x);
}
