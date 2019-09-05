package naivebayesalgorithm;

import java.util.ArrayList;

/**
 *
 * @author andy-
 */
public class NaiveBayes {
        
    /**
     * After training,
     * N contains the number of examples in each class
     * 
     * Indexed by class.
     */
    private int[] Nc;
    
    /**
     * After training,
     * Q contains the number of examples in each class divided by the total
     * number of examples in the training set.
     * 
     * Indexed by class.
     */
    private double[] Q;
    
    /**
     * After training,
     * F contains the number of examples in a specific class that match an 
     * attribute + 1, divided by the number of examples in that class + the
     * number of attributes.
     * 
     * Indexed by F[<class>][<attribute>][<attribute_bin>].
    */
    private double[][][] F;
    
    //private ArrayList<ArrayList<ArrayList<Double>>> yeet;
    
    private double[][][] classes;
    private double[][] attr;
    private double[] attr_values;
    
    public NaiveBayes() {
        
    }
    
    /**
     * Trains the algorithm with given training set(s), after which, you can
     * test further sets using test().
     * @param training_sets The training sets to train the algorithm with.
     */
    public void train(Set training_set) {
        //Find totals of examples in each class
        Example[] examples = training_set.getExamples();
        Nc = new int[training_set.numClasses()];
        for (Example example : examples) {
            Nc[example.getClassNum()]++;
        }
        //Find Q by dividing Nc by total number of examples
        Q = new double[Nc.length];
        for(int i = 0; i < Q.length; i++) {
            Q[i] = Nc[i] / training_set.numExamples();
        }
        
        //Initialize F
        //int max_bins = Arrays. IfsjdhflajhdfljksahfJKHDFKLJHDLJ
        int max_bins = 2;
        F = new double[training_set.numClasses()][training_set.numAttr()][max_bins];
        //Calculate F for each attribute in/for each class
        //Start by counting examples that match
        for(int c = 0; c < training_set.numClasses(); c++) { //Iterate through the classes
            for(Example e : examples) { //Find all examples in that class
                if(e.getClassnum() == c) {
                   for(int a = 0; a < e.getAttributes().length; a++) { //Access all the example's attributes
                       F[c][a][e.getAttributes()[a]]++; //Add one to the count of that attribute
                   }
                }
            }
        }
        //Finish calculating F by adding 1 and dividing by (#examples in the class + # attributes)
        for(int c = 0; c < training_set.numClasses(); c++) { //Iterate through the classes
            for(int a = 0; a < e.getAttributes().length; a++) { //Iterate through the attributes
                for(int a_value = 0; a_value < NUM_ATTR_BINS[a]; a_value++) { //Iterate through the possible values of the attributes
                    F[c][a][a_value] = (F[c][a][a_value] + 1) / (Nc[c] + ATTRIBUTES);
                }
            }
        }
        
        //Training complete!
    }
    
    /**
     * Tests the algorithm with given testing set(s) and outputs the data used
     * for loss metrics.
     * @param test_sets The test sets to test the algorithm with.
     */
    public void test(Set test_set) {
        Example[] examples = test_set.getExamples();
        for(Example example : examples) { //Classify each example in the set
            // C = Q * the product of all relevant F
            double product = 1;
            int[] attributes = example.getAttributes();
            //Find the product of all relevant F of each attribute
            for(int a = 0; a < attributes.length; a++) {
                product *= F[]
            }
        }
    }
    
}
