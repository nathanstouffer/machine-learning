/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayesalgorithm;

// import libraries
import java.util.ArrayList;

/**
 *
 * @author natha
 */
public class Set {
    
    // global variables to store set variables
    private final int num_classes;
    private final int num_attributes;
    private int num_examples = 0;
    // num_bins[0] corresponds to the number of bins for the 0th attribute of a class
    private int[] num_bins;
    // array storing class names. in our input files, 
    // classes are assigned to a number from 0 up to c,
    // the number of classes. The string can be
    // accessed using this array
    private String[] class_names;
    // ArrayList to store the examples in this subset
    // Arraylist with examples that make up the set
    private ArrayList<Example> examples = new ArrayList<Example>();
    
    /**
     * constructor to instantiate a Set out of just one subset
     * @param subset 
     */
    Set(Subset subset){
        // initialize global variables
        this.num_classes = subset.getNumClasses();
        this.num_attributes = subset.getNumAttributes();
        this.num_bins = subset.getNumBins();
        this.class_names = subset.getClassNames();
        this.num_examples = subset.getNumExamples();
        this.examples = subset.getExamples();
    }
    
    /**
     * constructor to instantiate a Set from an array of subsets
     * while excluding the subset at the index 'exclude'
     * @param subsets
     * @param exclude 
     */
    Set(Subset[] subsets, int exclude){
        // initialize global final variables
        this.num_classes = subsets[0].getNumClasses();
        this.num_attributes = subsets[0].getNumAttributes();
        this.num_bins = subsets[0].getNumBins();
        this.class_names = subsets[0].getClassNames();
        
        // ensure that the subset to be excluded is a valid subset
        if (exclude >= 0 && exclude < 10){
            // iterate through subsets
            for (int i = 0; i < 10; i++){
                // add subset to set if subset should not be excluded
                if (exclude != i){
                    // assign variable to current subset
                    Subset curr = subsets[i];
                    
                    // update num_examples
                    int curr_num_examples = curr.getNumExamples();
                    this.num_examples += curr_num_examples;
                    
                    // iterate through examples in a subset and add examples to the set
                    ArrayList<Example> to_add = curr.getExamples();
                    for (int j = 0; j < curr_num_examples; j++){
                        this.examples.add(to_add.get(j));
                    }
                }
            }
        }
        else{ System.out.println("exclude parameter must be 0 <= exclude < 10"); }
    }
        
    // getter methods
    public int getNumClasses(){ return this.num_classes; }
    public int getNumAttributes(){ return this.num_attributes; }
    public int getNumExamples(){ return this.num_examples; }
    public ArrayList<Example> getExamples(){ return this.examples; }
}
