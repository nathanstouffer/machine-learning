package naivebayesalgorithm;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @author andy-
 */
public class NaiveBayes {
        
    /**
     * After training,
     * Nc contains the number of examples in each class
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
    
    private int num_classes; // The number of classes contained in the set of examples
    private int num_attributes; // The number of attributes that each example has
    private int[] num_attribute_bins; //Indexed by attribute, the number of possible integer values that attribute can take on
    
    public NaiveBayes() {
        
    }
    
    /**
     * Trains the algorithm with given training set, after which, you can
     * test further sets using test(). Note: Combine multiple subsets into one
     * set before using the train() method.
     * @param training_set The training sets to train the algorithm with.
     */
    public void train(Set training_set) {
        // Initialize values
        num_classes = training_set.getNumClasses();
        num_attributes = training_set.getNumAttributes();
        num_attribute_bins = training_set.getNumBins();
        //Find totals of examples in each class
        ArrayList<Example> examples = training_set.getExamples();        
        Nc = new int[num_classes];
        for(Example example : examples) {
            Nc[example.getClassType()]++;
        }
        //Find Q by dividing Nc by total number of examples
        Q = new double[Nc.length];
        for(int i = 0; i < Q.length; i++) {
            Q[i] = Nc[i] / training_set.getNumExamples();
        }
        
        //Initialize F storage
        // Create a multidimensional array to hold values of F. See global variable description for more.
        F = new double[num_classes][num_attributes][findMax(num_attribute_bins)];
        //Calculate F for each attribute in/for each class
        //Start by counting examples that match
        for(int c = 0; c < num_classes; c++) { //Iterate through the classes
            for(Example e : examples) { //Find all examples in that class
                if(e.getClassType() == c) {
                   for(int a = 0; a < num_attributes; a++) { //Access all the example's attributes
                       F[c][a][e.getAttributes().get(a)]++; //Add one to the count of that attribute
                   }
                }
            }
        }
        //Finish calculating F by adding 1 and dividing by (#examples in the class + # attributes)
        for(int c = 0; c < num_classes; c++) { //Iterate through the classes
            for(int a = 0; a < num_attributes; a++) { //Iterate through the attributes
                for(int a_value = 0; a_value < num_attribute_bins[a]; a_value++) { //Iterate through the possible values of the attributes
                    F[c][a][a_value] = (F[c][a][a_value] + 1) / (Nc[c] + num_attributes);
                }
            }
        }
        
        System.out.println("Naive Bayes trained with " + training_set.getNumExamples() + " examples.");
        //Training complete!
    }
    
    /**
     * Tests the algorithm with given testing set(s) and outputs the data used
     * for loss metrics.
     * @param test_set The test sets to test the algorithm with.
     * @return int[] An array, indexed by example in the test set, with integers
     * representing their classifications.
     */
    public int[] test(Set test_set) {
        // Initialize array to hold the resulting classifications of the test set examples
        int[] classification = new int[test_set.getNumExamples()];
        
        //Begin testing
        ArrayList<Example> examples = test_set.getExamples();
        for(Example example : examples) { //Classify each example in the set
            ArrayList<Integer> attributes = example.getAttributes();
            double[] Cs = new double[num_classes];
            //Calculate C of each class. C = Q * the product of all relevant F
            for(int classid = 0; classid < num_classes; classid++) {
                int product = 1;
                //Find the product of all relevant F of each attribute
                for(int attrid = 0; attrid < num_attributes; attrid++) {
                    product *= F[classid][attrid][attributes.get(attrid)];
                }
                Cs[classid] = Q[classid] * product;
            }
            //Choose the class based on the maximum value of C in Cs.
            int ex_class = 0;
            for(int i = 1; i < Cs.length; i++) {
                if(Cs[i] > Cs[ex_class]){
                    ex_class = i;
                }
            }
            System.out.println("Class is: " + ex_class); //OUTPUT CLASSIFICATION
        }
        
        System.out.println("Naive Bayes tested with " + test_set.getNumExamples() + " examples.");
        return classification;
    }
    
    /**
     * Used to find the max of an int array with positive nonzero values.
    */
    private int findMax(int[] arr) {
        int max = 0;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }
    
}
