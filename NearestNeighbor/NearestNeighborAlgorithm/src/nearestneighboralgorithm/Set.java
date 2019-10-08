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
    private final int num_attributes;
    private final int num_classes;
    // array storing class names. in our input files, classes are assigned to
    // a number from 0 up to c, the number of classes. The string can be
    // accessed using this array
    private String[] class_names;
    // ArrayList to store the examples in this subset
    // Arraylist with examples that make up the set
    private ArrayList<Example> examples = new ArrayList<Example>();
    
    /**
     * constructor to instantiate an empty set object
     * @param num_classes
     * @param num_attributes
     * @param class_names 
     */
    Set(int num_attributes, int num_classes, String[] class_names){
        // populate final global variables
        this.num_attributes = num_attributes;
        this.num_classes = num_classes;
        this.class_names = class_names;
    }
    
    /**
     * constructor to instantiate a Set from an array of subsets
     * while excluding the subset at the index 'exclude'. This constructor
     * should be used when creating a single set to train with in 10-fold
     * cross validation.
     * 
     * if validation_set is passed as true, the constructor assumes
     * that the 0th element in subsets will be used as a validation set
     * 
     * @param subsets
     * @param exclude 
     * @param validation_set
     */
    Set(Set[] subsets, int exclude, boolean validation_set){
        // initialize global final variables
        this.num_attributes = subsets[0].getNumAttributes();
        this.num_classes = subsets[0].getNumClasses();
        this.class_names = subsets[0].getClassNames();
        
        // ensure that the subset to be excluded is a valid subset
        boolean valid_index = false;
        if (validation_set){
            // we will use subsets[0] as the validation set
            // therefore it is not used in 10-fold cross validation
            if (exclude >= 1 && exclude < subsets.length){ valid_index = true; }
        }
        else if (exclude >= 0 && exclude < subsets.length){ valid_index = true; }
        
        if (valid_index){
            // i represents the index in subsets
            // i begins at the 0th element in subsets
            int i = 0;
            // ignore the 0th element in subsets if a validation set is needed
            if (validation_set){ i = 1; }
            // iterate through subsets
            for ( ; i < subsets.length; i++){
                // add subset to set if subset should not be excluded
                if (exclude != i){
                    // assign curr to current subset
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
     * method to return a clone of the object 'this'
     * @param orig
     * @return 
     */
    public Set clone(){
        // initialize new Set object
        Set clone = new Set(this.num_attributes, this.num_classes, this.class_names);
        
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
     * overloaded method to delete the ith example in a set
     * @param index 
     */
    public void delExample(int index){ this.examples.remove(index); }
    
    /**
     * overloaded method to delete an example in a set
     * @param ex 
     */
    public void delExample(Example ex){ this.examples.remove(ex); }
    
    public void replaceExample(int index, Example ex){ this.examples.set(index, ex); }
    
    public void clearSet(){ this.examples.clear(); }
    
    /**
     * method to return an iterator over the examples in the Set
     * @return 
     */
    @Override
    public Iterator<Example> iterator() { return examples.iterator(); }
    
    // getter methods
    public Example getExample(int index){ return this.examples.get(index); }
    public int getNumClasses(){ return this.num_classes; }
    public int getNumAttributes(){ return this.num_attributes; }
    public int getNumExamples(){ return this.examples.size(); }
    public String[] getClassNames(){ return this.class_names; }
    // outdated getter method to return examples
    // we now use iterator interface
    public ArrayList<Example> getExamples(){ return this.examples; }

}
