/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package networklayer;

import datastorage.Example;
import java.util.Random;
import java.util.ArrayList;

/**
 * class to store a vector that can be manipulated by the
 * neural network
 * 
 * A Vector can be added with another Vector or sent in to a
 * Matrix to be multiplied
 * 
 * @author natha
 */
public class Vector {
    
    // array to store the values of the Vector
    private double[] vals;
    
    /**
     * constructor to create a correctly sized vector
     * where all values are 0.0
     * 
     * @param length 
     */
    public Vector(int length){
        // initialize array to correct size
        this.vals = new double[length];
        // set all values to 0.0
        for (int i = 0; i < length; i++) { this.vals[i] = 0.0; }
    }
    
    /**
     * constructor to build a Vector from an Example
     * 
     * The result automatically includes a bias multiplier
     * 
     * @param ex 
     */
    public Vector(Example ex){
        // get attributes
        ArrayList<Double> attr = ex.getAttributes();
        // instantiate vals to correct size
        this.vals = new double[attr.size() + 1];
        // add in bias term for dot product
        this.vals[0] = 1.0;
        
        // add each attribute to the vector
        for (int i = 0; i < attr.size(); i++) { this.set(i+1, attr.get(i)); }
    }
    
    /**
     * method to insert 1.0 to the first position of a vector
     * this is used to multiply the bias by 1.0
     */
    public void insertBiasMultiplier() {
        // clone the array
        double[] clone = this.vals.clone();
        
        // instantiate array to new size
        this.vals = new double[this.getLength()+1];
        // insert bias multiplier
        this.set(0, 1.0);
        
        // iterate through clone, copying into new vector
        for (int i = 0; i < clone.length; i++) { this.set(i+1, clone[i]); }
    }
    
    /**
     * method to add two Vectors together, the result is stored in the first vector
     * 
     * the two vectors must be of the same dimension, otherwise an error
     * message is printed
     * 
     * @param to_add 
     */
    public void plusEquals(Vector to_add) {
        // make sure Vectors are of the same dimension
        if (this.getLength() != to_add.getLength()){ System.err.println("Vectors have different dimensions"); }
        else {
            // add the two vectors
            for (int i = 0; i < this.getLength(); i++) {
                // compute new value
                double val = this.get(i) + to_add.get(i);
                // update new value in the array
                this.set(i, val);
            }
        }
    }
    
    /**
     * method to randomly populate the vector with values
     * between lower and upper
     * 
     * @param lower
     * @param upper 
     */
    protected void randPopulate(double lower, double upper) {
        // instantiate random generator
        Random rand = new Random();
        
        // populate the array with random values between bounds
        for (int i = 0; i < this.getLength(); i++) {
            // generate double
            double val = rand.nextDouble();
            val = lower + (upper - lower) * val;
            this.set(i, val);
        }
    }
    
    /**
     * method to compute the dot product between two vectors
     * this returns a real value, the result of a dot product
     * 
     * the two vectors of the same dimension, otherwise an 
     * error message is printed
     * 
     * @param vec
     * @return 
     */
    protected double dotProd(Vector to_prod) {
        // make sure vectors are of the same dimension
        if (this.vals.length != to_prod.getLength()) {
            System.err.println("Vectors have different dimensions");
            return Double.NaN;
        }
        else {
            double prod = 0.0;
            // compute dot product
            for (int i = 0; i < this.getLength(); i++) { prod += this.get(i) * to_prod.get(i); }
            
            // return the result
            return prod;
        }
    }
    
    /**
     * method to multiply each value in the Vector by the multiplier
     * @param multiplier 
     */
    protected void timesEquals(double multiplier) {
        // iterate through vector
        for (int i = 0; i < this.getLength(); i++) { this.set(i, multiplier * this.get(i)); }
    }
    
    /**
     * method to divide each value in the Vector by the divisor
     * @param divisor 
     */
    protected void divEquals(double divisor) {
        // iterate through vector
        for (int i = 0; i < this.getLength(); i++) { this.set(i, this.get(i) / divisor); }
    }
  
    /**
     * method to find the index of the largest element in the Vector
     * @return The index of the largest element.
     */
    public int getMaxIndex() {
        int max = 0;
        for(int i = 0; i < vals.length; i++) {
            if( vals[i] > vals[max]) {
                max = i;
            }
        }
        return max;
    }
    
    public void set(int index, double val) { this.vals[index] = val; }
    public double get(int index) { return this.vals[index]; }
    public int getLength() { return this.vals.length; }
    
}
