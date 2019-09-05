package naivebayesalgorithm;

/**
 *
 * @author andy-
 */
public class NaiveBayes {
        
    /**
     * After training,
     * Q contains the number of examples in each class divided by the total
     * number of examples in the training set.
     */
    private double[] Q;
    
    /**
     * After training,
     * F contains the number of examples in a specific class that match an 
     * attribute + 1, divided by the number of examples in that class + the
     * number of attributes
    */
    
    
       
    public NaiveBayes() {
        
    }
    
    /**
     * Trains the algorithm with given training set(s), after which, you can
     * test further sets using test().
     * @param training_sets The training sets to train the algorithm with.
     */
    public void train(Subset[] training_sets, int num_class, int num_attr) {
        table = new int[num_class][num_attr]; //Initialize table used for algorithm with all zeros
        for(int i = 0; i < training_sets.length; i++) { //Train with each set
            
        }
    }
    
    /**
     * Tests the algorithm with given testing set(s) and outputs the data used
     * for loss metrics.
     * @param test_sets The test sets to test the algorithm with.
     */
    public void test(Subset[] test_sets) {
        for(int i = 0; i < test_sets.length; i++) { //Test with each set
            
        }
    }
    
}
