/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets.layer;

import datastorage.Example;
import datastorage.SimilarityMatrix;
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
        for (int i = 0; i < length; i++) { this.set(i, 0.0); }
    }
    
    /**
     * constructor to build a Vector from an Example
     * 
     * @param ex 
     */
    public Vector(Example ex, SimilarityMatrix[] sim){
        // get attributes
        ArrayList<Double> attr = ex.getAttributes();
        int size = attr.size();
        for (int i = 0; i < sim.length; i++) {
            // add the number of categorical options to the size 
            // of the array (subracting one for the attribute already
            // in the example)
            size += sim[i].getNumOptions() - 1;
        }
        // instantiate vals to correct size
        this.vals = new double[size];
        
        // add each attribute to the vector
        int s = 0;      // index of similarity matrix
        int a = 0;      // index of attributes
        for (int i = 0; i < this.vals.length; i++) { 
            // check for indexing error
            if (s < sim.length) {
                // test if attribute is categorical
                if (a == sim[s].getAttrIndex()) {
                    double val = attr.get(a);
                    a++;
                    // iterate through options
                    int j = 0;
                    for (j = 0; j < sim[s].getNumOptions(); j++) {
                        // test if value of attribute is the current option
                        if ((int)val == j) { this.set(i + j, 1.0); }
                        else { this.set(i + j, 0.0); }
                    }
                    // set i to appropriate value
                    i += j - 1;
                    // increment s
                    s++;
                }
                // otherwise, just add the value
                else { 
                    this.set(i, attr.get(a)); 
                    a++;
                }
            }
            // otherwise just add the value
            else { 
                this.set(i, attr.get(a)); 
                a++;
            }
        }
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
     * public method to add two vectors and return the diff
 
 Neither input is changed in this process
     * 
     * @param to_add
     * @return 
     */
    public Vector plus(Vector to_add) {
        // check if Vectors have same dimension
        if (this.getLength() != to_add.getLength()){ 
            System.err.println("Vectors have different dimensions"); 
            return null;
        }
        else {
            // create new Vector
            Vector sum = new Vector(this.getLength());
            for (int s = 0; s < sum.getLength(); s++) { 
                sum.set(s, this.get(s) + to_add.get(s)); 
            }
            return sum;
        }
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
     * public method to subtract two vectors and return the difference
     * 
     * Neither input is changed in this process
     * 
     * @param to_sub
     * @return 
     */
    public Vector minus(Vector to_sub) {
        // check if Vectors have same dimension
        if (this.getLength() != to_sub.getLength()){ 
            System.err.println("Vectors have different dimensions"); 
            return null;
        }
        else {
            // create new Vector
            Vector diff = new Vector(this.getLength());
            for (int s = 0; s < diff.getLength(); s++) { 
                diff.set(s, this.get(s) - to_sub.get(s)); 
            }
            return diff;
        }
    }
    
    /**
     * public method to multiply a vector by a multiplier.
     * A new vector is returned
     * @param multiplier
     * @return 
     */
    public Vector times(double multiplier) {
        // instantiate new Vector
        Vector prod = new Vector(this.getLength());
        for (int p = 0; p < prod.getLength(); p++) {
            prod.set(p, multiplier * this.get(p));
        }
        return prod;
    }
    
    /**
     * method to multiply each value in the Vector by the multiplier
     * @param multiplier 
     */
    public void timesEquals(double multiplier) {
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
     * method to clear the Vector
     * inserts 0.0 into every value in the Vector
     */
    protected void clear() { 
        for (int i = 0; i < this.getLength(); i++) { this.set(i, 0.0); } 
    }
  
    /**
     * method to find the index of the largest element in the Vector
     * @return The index of the largest element.
     */
    public int getMaxIndex() {
        int max = 0;
        for(int i = 0; i < vals.length; i++) {
            if(vals[i] > vals[max]) {
                max = i;
            }
        }
        return max;
    }
    
    public Vector clone() {
        Vector clone = new Vector(this.getLength());
        for (int i = 0; i < clone.getLength(); i++) { clone.set(i, this.get(i)); }
        return clone;
    }
    
    public void set(int index, double val) { this.vals[index] = val; }
    public double get(int index) { return this.vals[index]; }
    public int getLength() { return this.vals.length; }
    
    @Override
    public String toString() {
        String out = "[";
        for(int i = 0; i < vals.length-1; i++) {
            out += "" + vals[i] + ",";
        }
        out += "" + vals[vals.length-1] + "]";
        return out;
    }
    
}
