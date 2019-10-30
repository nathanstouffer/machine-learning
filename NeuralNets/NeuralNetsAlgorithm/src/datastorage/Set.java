package datastorage;

// import libraries
import java.util.ArrayList;
import java.util.Random;

/**
 * Class that represents a set of examples from a dataset
 * Set can be constructed with no values or from existing sets
 * @author natha
 */
public class Set implements Cloneable {
    
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
    public Set(int num_attributes, int num_classes, String[] class_names){
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
     * if need_validation_set is passed as true, the constructor assumes
     * that the 0th element in subsets will be used as a validation set
     * 
     * @param subsets
     * @param exclude 
     * @param need_validation_set
     */
    public Set(Set[] subsets, int exclude, boolean need_validation_set){
        // initialize global final variables
        this.num_attributes = subsets[0].getNumAttributes();
        this.num_classes = subsets[0].getNumClasses();
        this.class_names = subsets[0].getClassNames();
        
        // ensure that the subset to be excluded is a valid subset
        boolean valid_index = false;
        if (need_validation_set){
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
            if (need_validation_set){ i = 1; }
            // iterate through subsets
            for ( ; i < subsets.length; i++){
                // add subset to set if subset should not be excluded
                if (exclude != i){
                    // assign curr to current subset
                    Set curr = subsets[i];
                    
                    // iterate through examples in a subset and add examples to the set
                    for (int j = 0; j < curr.getNumExamples(); j++){ this.addExample(curr.getExample(j)); }
                    
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
    @Override
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
     * method to add an example at index
     * @param index
     * @param ex 
     */
    public void addExample(int index, Example ex){ this.examples.add(index, ex); }
    
    /**
     * overloaded method to delete the ith example in a set
     * @param index 
     */
    public void rmExample(int index){ this.examples.remove(index); }
    
    /**
     * overloaded method to delete an example in a set
     * @param ex 
     */
    public void rmExample(Example ex){ this.examples.remove(ex); }
    
    public void replaceExample(int index, Example ex){ this.examples.set(index, ex); }
    
    public void clearSet(){ this.examples.clear(); }
    
    /**
    * Make random batches out of the current set
     * @param batch_size The percentage of examples to be put in each batch.
    * @return Set[] The randomized batches to be used as desired.
    */
    public Set[] getRandomBatches(double batch_size) {
        // Clone current set so that we can randomize it
        Set clone = clone();
        // Initialize new batches
        Set[] batches = new Set[(int)Math.ceil(1/batch_size) + 1];
        for(int i = 0; i < batches.length; i++) {
            batches[i] = new Set(getNumAttributes(), getNumClasses(), getClassNames());
        }
        int current_batch = 0;
        // Fill the batches
        for(int i = 0; i < getNumExamples(); i++) {
            // Generate random index to put in the batch
            Random rand = new Random();
            int rand_index = rand.nextInt(clone.getNumExamples());
            batches[current_batch].addExample(clone.getExample(rand_index));
            clone.rmExample(rand_index);
            // Increment to next batch if necessary
            if ( (i+1) % (int)(getNumExamples() * batch_size) == 0) {
                current_batch++;
            }
        }
        return batches;
    }
    
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
