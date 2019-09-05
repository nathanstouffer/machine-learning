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
public class Subset {
    
    // variables to store subset information
    private final int num_classes;
    private final int num_attributes;
    private int num_examples;
    // array storing the number of bins for a corresponding attribute
    // num_bins[0] corresponds to the number of bins for the 0th attribute of a class
    private int[] num_bins;
    // array storing class names. in our input files, 
    // classes are assigned to a number from 0 up to c,
    // the number of classes. The string can be
    // accessed using this array
    private String[] class_names;
    // ArrayList to store the examples in this subset
    private ArrayList<Example> examples = new ArrayList<Example>();
    
    Subset(int num_classes, int num_attributes, int[] num_bins, String[] class_names){
        this.num_classes = num_classes;
        this.num_attributes = num_attributes;
        this.num_bins = num_bins;
        this.class_names = class_names;
    }
    
    public void addExample(Example ex){
        this.examples.add(ex);
        this.num_examples++;
    }
    
    // getter methods
    public int getNumClasses(){ return this.num_classes; }
    public int getNumAttributes(){ return this.num_attributes; }
    public int getNumExamples(){ return this.num_examples; }
    public int[] getNumBins(){ return this.num_bins; }
    public String[] getClassNames(){ return this.class_names; }
    public ArrayList<Example> getExamples(){ return this.examples; }
    
}
