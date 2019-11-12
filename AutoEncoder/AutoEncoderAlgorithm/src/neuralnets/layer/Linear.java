/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets.layer;

/**
 *
 * @author natha
 */
public class Linear implements IActFunct {
    
    // Vector to store the derivative of each value
    // in the Vector
    private Vector deriv;
    
    public Linear() { this.deriv = null; }
    
    /**
     * method to compute the activation of each value in
     * an output Vector. The activation function used is
     * a linear sum
     * 
     * this method acts directly on the Vector sent in
     * as an argument, so there is no need to return a
     * new Vector
     * 
     * activation should be computed on the output of an
     * entire layer
     * 
     * @param vec
     * @return 
     */
    @Override
    public Vector computeAct(Vector vec) {
        Vector activ = new Vector(vec.getLength());
        for (int i = 0; i < activ.getLength(); i++) {
            // the dot product already computes a linear sum
            // therefore no activation computation is required
            activ.set(i, vec.get(i));
        }
        
        // instantiate and populate deriv
        this.deriv = new Vector(vec.getLength());
        for (int i = 0; i < this.deriv.getLength(); i++) {
            // since the activation is a linear sum, the derivative
            // at each value is just 1.0
            this.deriv.set(i, 1.0);
        }
        
        // return activation Vector
//        System.out.println("ACTIVATION VECTOR " + activ);
        return activ;
    }
    
    /**
     * method to return the derivative Vector
     * the values in this vector are computed from the
     * values sent in to the computeAct method
     * @param vec
     * @return 
     */
    @Override
    public Vector getDeriv() {
        // check if deriv is null
        if (this.deriv == null) { System.err.println("Derivative has not been computed."); return null; }
        // otherwise, return deriv
        else { return this.deriv; }
    }
    
}
