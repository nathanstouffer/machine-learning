package nearestneighboralgorithm;

// import libraries
import java.util.Iterator;
import java.util.ArrayList;

/**
 * Class that represents a set of examples from a dataset
 * Set can be constructed with no values or from existing sets
 * @author natha
 */
public class Set implements Iterable<Example>, Cloneable {
    
    // global variables to store set variables
    private final int num_classes;
    private final int num_attributes;
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
     * constructor to instantiate an empty set object
     * @param num_classes
     * @param num_attributes
     * @param num_bins
     * @param class_names 
     */
    Set(int num_classes, int num_attributes, int[] num_bins, String[] class_names){
        this.num_classes = num_classes;
        this.num_attributes = num_attributes;
        this.num_bins = num_bins;
        this.class_names = class_names;
    }
    
    /**
     * constructor to instantiate a Set from an array of subsets
     * while excluding the subset at the index 'exclude'
     * @param subsets
     * @param exclude 
     */
    Set(Set[] subsets, int exclude){
        // initialize global final variables
        this.num_classes = subsets[0].getNumClasses();
        this.num_attributes = subsets[0].getNumAttributes();
        this.num_bins = subsets[0].getNumBins();
        this.class_names = subsets[0].getClassNames();
        
        // ensure that the subset to be excluded is a valid subset
        if (exclude >= 0 && exclude < subsets.length){
            // iterate through subsets
            for (int i = 0; i < subsets.length; i++){
                // add subset to set if subset should not be excluded
                if (exclude != i){
                    // assign variable to current subset
                    Set curr = subsets[i];
                    
                    // iterate through examples in a subset and add examples to the set
                    for (Example ex: curr){ this.addExample(ex); }
                    
                    // outdated code before iterable interface
                    /*int curr_num_examples = curr.getNumExamples();
                    
                    // iterate through examples in a subset and add examples to the set
                    ArrayList<Example> to_add = curr.getExamples();
                    for (int j = 0; j < curr_num_examples; j++){
                        this.addExample(to_add.get(j));
                    } 
                    */
                }
            }
        }
        else{ System.out.println(String.format("exclude argument must be 0 <= exclude < %d", subsets.length)); }
    }
    
    /**
     * method to return a clone of the object this method is called from
     * @param orig
     * @return 
     */
    public Set clone(){
        // initialize new Set object
        Set clone = new Set(this.num_classes, this.num_attributes, this.num_bins, this.class_names);
        
        // add each example in this to clone
        for (Example ex: examples){ clone.addExample(ex); }
        
        return clone;
    }
    
    /**
     * method to add example to examples ArrayList
     * @param ex 
     */
    public void addExample(Example ex){ this.examples.add(ex); }
    
    /**
     * overloaded method to delete an example in a set
     * @param index 
     */
    public void delExample(int index){ this.examples.remove(index); }
    
    /**
     * overloaded method to delete an example in a set
     * @param ex 
     */
    public void delExample(Example ex){ this.examples.remove(ex); }
    
    /**
     * method to return an iterator over the examples in the Set
     * @return 
     */
    @Override
    public Iterator<Example> iterator() { return examples.iterator(); }
    
    // getter methods
    public int getNumClasses(){ return this.num_classes; }
    public int getNumAttributes(){ return this.num_attributes; }
    public int getNumExamples(){ return this.examples.size(); }
    public int[] getNumBins(){ return this.num_bins; }
    public String[] getClassNames(){ return this.class_names; }
    // outdated getter method to return examples
    // we now use iterator interface
    // public ArrayList<Example> getExamples(){ return this.examples; }

}
