/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package datastorage;

/**
 * Class to represent a Similarity Matrix. 
 * 
 * Each matrix consists of its attribute and a 2-dim
 * array that stores appropriate probabilities
 * 
 * An object of this type should be used for computing
 * distances between categorical variables
 *
 * @author natha
 */
public class SimilarityMatrix {
    
    // global variables
    private int attr_index;
    private int num_options;
    private int num_classes;
    
    // this is the similarity matrix
    // it is a 2-dim array accessed first by the attribute value, then the class
    private double[][] matrix; //= new double[this.num_options][this.num_classes]; ------BRUH STOUFF
    
    public SimilarityMatrix(int attr_index, int num_options, int num_classes){
        this.attr_index = attr_index;
        this.num_options = num_options;
        this.num_classes = num_classes;
        
        matrix = new double[this.num_options][this.num_classes];
    }
    
    /**
     * method to add a row to the similarity matrix
     * each row should consist of probabilities that the value 'value'
     * appears in the kth class in the variable 'line'
     * @param value
     * @param line 
     */
    public void addRow(int option, String line){
        // convert the input line to a string array
        String[] data = line.split(",");
        // insert values into the similarity matrix
        for (int i = 0; i < data.length; i++){ matrix[option][i] = Double.parseDouble(data[i]); }
    }
    
    /**
     * method to returns the probability that the option appears
     * in the class 'classification'
     * @param option
     * @param classification
     * @return 
     */
    public double getProb(int option, int classification){ return this.matrix[option][classification]; }
    
    // getter methods
    public int getAttrIndex(){ return this.attr_index; }
    public int getNumClasses(){ return this.num_classes; }
    public int getNumOptions() { return this.num_options; }
    
}
