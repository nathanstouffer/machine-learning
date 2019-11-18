/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets.layer;

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

    /**
     * constructor to create a new, empty layer
     * 
     * @param act_funct
     * @param num_outputs
     * @param num_inputs 
     */
    public Layer(IActFunct act_funct, int num_outputs, int num_inputs) {
        this.act_funct = act_funct;
        this.weights = new Matrix(num_outputs, num_inputs);
        this.deriv = null;
    }
    
    /**
     * constructor to create a layer from an existing weights matrix
     * 
     * This will typically be read in from a file
     * @param act_funct
     * @param weights 
     */
    public Layer(IActFunct act_funct, Matrix weights) {
        this.act_funct = act_funct;
        this.weights = weights;
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
//        System.out.println("FEED FORWARD VAL " + output);
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
     * method to add a Matrix to the weights in the current layer
     * @param to_add must be a Matrix
     */
    public void plusEquals(Matrix to_add) { this.weights.plusEquals(to_add); }

    /**
     * method to randomly populate weights in a layer
     * @param lower
     * @param upper
     */
    public void randPopulate(double lower, double upper) { this.weights.randPopulate(lower, upper); }

    public String toString() {
        String output = this.act_funct.toString() + ",";
        output += Integer.toString(this.weights.getNumRows()) + ",";
        output += Integer.toString(this.weights.getNumCol()) + ",\n";
        
        // iterate through rows
        for (int i = 0; i < this.weights.getNumRows(); i++) {
            Vector row = this.weights.getRow(i);
            // iterate through the row
            for (int j = 0; j < this.weights.getNumCol(); i++) {
                output += Double.toString(row.get(j)) + ",";
            }
            output += "\n";
        }
        return output;
    }
    
    public void delRow(int index) { this.weights.delRow(index); }
    public void delCol(int index) { this.weights.delCol(index); }
    
    // change these method names
    public int getNumNodes() { return this.weights.getNumRows(); }
    public int getNumInputs() { return this.weights.getNumCol(); }
    
    public Matrix getWeights() { return this.weights; }
}
