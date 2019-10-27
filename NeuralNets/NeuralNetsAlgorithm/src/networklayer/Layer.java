/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package networklayer;

/**
 * class that represents a layer in a neural network
 * 
 * This class consists of a weight matrix and an activation function
 * 
 * @author natha
 */
public class Layer {
    
    // matrix to store the weights in a layer
    private Matrix weights;
    // activation function used on this layer
    private IActFunct act_funct;
    // derivative of each value in the output vector
    private Vector deriv;
    
    public Layer(IActFunct act_funct, int num_rows, int num_col) {
        this.act_funct = act_funct;
        this.weights = new Matrix(num_rows, num_col);
        this.deriv = null;
    }
    
    /**
     * method to feed an input forward
     * @param input
     * @return 
     */
    public Vector feedForward(Vector input) {
        // compute matrix-vector mult
        Vector output = this.weights.mult(input);
        // compute activation
        output = this.act_funct.computeAct(output);
        // get derivation
        this.deriv = this.act_funct.getDeriv();
        // return output
        return output;
    }
    
    /**
     * method to return the derivative Vector
     * the values in this vector are computed from the
     * values sent in to the computeAct method
     * @param vec
     * @return 
     */
    public Vector getDeriv() {
        // check if deriv is null
        if (this.deriv == null) { System.err.println("Derivative has not been computed."); return null; }
        // otherwise, return deriv
        else { return this.deriv; }
    }
    
    /**
     * method to randomly populate weights in a layer
     * @param lower
     * @param upper 
     */
    public void randPopulate(double lower, double upper) { this.weights.randPopulate(lower, upper); }
    
    public int getNumRows() { return this.weights.getNumRows(); }
    public int getNumCol() { return this.weights.getNumColumns(); }
}
