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
public class Logistic implements IActFunct {
    
    // Vector to store the derivative of each value
    // in the Vector
    private Vector deriv;
    
    // empty constructor
    public Logistic() { this.deriv = null; }

    /**
     * method to compute the activation of each value in
     * an output Vector. The activation function used is
     * the logistic function
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
        // compute sigmoid of each value in vec
        for (int i = 0; i < activ.getLength(); i++) {
            // get original value
            double orig = vec.get(i);
            // compute logistic function
            activ.set(i, this.logisticFunct(orig));
        }
        
        // instantiate and populate deriv
        this.deriv = new Vector(activ.getLength());
        for (int i = 0; i < this.deriv.getLength(); i++) {
            // original val
            double val = activ.get(i);        // logistic function has already been applied
            // compute derivative
            val = val * (1 - val);
            // store in deriv
            this.deriv.set(i, val);
        }
        
        // return activation 
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
    
    /**
     * method to compute the logistic function of a value
     * @param orig
     * @return 
     */
    private double logisticFunct(double orig) {
        double exp = Math.exp(-1 * orig);
        return 1 / (1 + exp);
    }
    
}
