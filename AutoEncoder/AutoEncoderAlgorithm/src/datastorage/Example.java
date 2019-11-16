/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package datastorage;

// import libraries
import java.util.ArrayList;
import neuralnets.layer.Vector;

/**
 * Class that represents one example in a dataset
 * Each example has a value, a subset that it belongs to,
 * and a list of attributes
 * 
 * @author natha
 */
public class Example {
    
    // global variable to store the type of class an example is in
    private final double value;
    // global variable to store which subset an example belongs to
    private final int subset_index;
    // global array to store values of each attribute for the example
    private ArrayList<Double> attr;
    
    /**
     * Constructor to populate global variables
     * @param input
     * @param num_attr 
     */
    public Example(String line, int num_attr){
        // initialize size of attr array
        this.attr = new ArrayList<Double>(num_attr);

        // split the input line into a String array
        String[] data = line.split(",");
        
        // populate class type and subset_index;
        this.value = Double.parseDouble(data[0]);
        this.subset_index = Integer.parseInt(data[1]);
        
        // populate attr ArrayList with attribute values
        for (int i = 0; i < num_attr; i++){ this.attr.add(Double.parseDouble(data[i+2])); }
    }
    
    /**
     * constructor used to create an example from a list of attributes
     * @param value
     * @param attributes 
     */
    public Example(double value, ArrayList<Double> attributes){
        this.attr = attributes;
        this.value = value;
        this.subset_index = -1;
    }
    
    /**
     * constructor to create an example from a Vector which is the values
     * of the encoded layer of an Auto encoder
     * @param value
     * @param subset_index
     * @param vec 
     */
    public Example(double value, int subset_index, Vector vec) {
        this.value = value;
        this.subset_index = subset_index;
        this.attr = new ArrayList<Double>();
        
        for (int i = 0; i < vec.getLength(); i++) {
            this.attr.add(vec.get(i));
        }
    }
    
    /**
     * constructor to create an example from a Vector, which
     * is the reconstructed output of an Auto-Encoder
     * @param value
     * @param subset_index
     * @param vec
     * @param sim
     */
    public Example(double value, int subset_index, Vector vec, SimilarityMatrix[] sim) {
        this.value = value;
        this.subset_index = subset_index;
        this.attr = new ArrayList<Double>();
        
        int a = 0;      // index of attributes
        int s = 0;      // index of similarity matrix
        // iterate through output vector
        for (int i = 0; i < vec.getLength(); a++) { 
            // test if we are beyond categorical attributes
            if (s >= sim.length) { this.attr.add(vec.get(i)); i++; }
            else {
                // check if current attribute is not categorical
                if (a != sim[s].getAttrIndex()) { this.attr.add(vec.get(i)); i++; }
                else {
                    // current attribute is categorical
                    // we then select the option with the highest probability
                    double[] options = new double[sim[s].getNumOptions()];
                    for (int j = 0; j < options.length; j++) {
                        options[j] = vec.get(i);
                        i++;
                    }
                    double max = (double)this.computeMaxIndex(options);
                    this.attr.add(max);
                }
            }
        }
    }
    
    /**
     * method to return a string representation of an example
     * @return 
     */
    public String toString() {
        String val = "------------ EXAMPLE ------------\n";
        val += "VALUE: " + this.value + "\n";
        val += "ATTRIBUTES: [ " + this.attr.get(0);
        for (int i = 1; i < this.attr.size(); i++) {
            val += ", " + this.attr.get(i);
        }
        val += "]";
        return val;
    }
    
    /**
     * private method to compute the index of the maximum value
     * in an array
     * @param arr
     * @return 
     */
    private int computeMaxIndex(double[] arr) {
        int max = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[max]) { max = i; }
        }
        return max;
    }
    
    // getter methods
    public double getValue(){ return this.value; }
    public int getSubsetIndex(){ return this.subset_index; }
    public ArrayList<Double> getAttributes(){ return this.attr; }
    
}
