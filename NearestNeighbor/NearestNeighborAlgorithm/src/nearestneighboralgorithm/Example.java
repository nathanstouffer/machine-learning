/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

// import libraries
import java.util.ArrayList;

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
    Example(String line, int num_attr){
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
    
    Example(ArrayList<Double> attributes){
        this.attr = attributes;
        this.value = 0.0;
        this.subset_index = 0;
    }
    
    // getter methods
    public double getValue(){ return this.value; }
    public int getSubsetIndex(){ return this.subset_index; }
    // method to return a clone of the attributes
    // a client of an Example object should only be able to view the information, not edit
    public ArrayList<Double> getAttributes(){ return (ArrayList<Double>)this.attr.clone(); }
    
}
