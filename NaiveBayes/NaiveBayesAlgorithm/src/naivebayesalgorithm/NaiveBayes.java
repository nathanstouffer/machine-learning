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
    
    private int num_classes;
    private int num_attributes;
    private int[] num_attribute_bins;
    
    public NaiveBayes() {
        
    }
    
    /**
     * Trains the algorithm with given training set(s), after which, you can
     * test further sets using test().
     * @param training_set The training sets to train the algorithm with.
     */
    public void train(Set training_set) {
        //Find totals of examples in each class
        ArrayList<Example> examples = training_set.getExamples();        
        Nc = new int[training_set.getNumClasses()];
        for (Example example : examples) {
            Nc[example.getClassType()]++;
        }
        //Find Q by dividing Nc by total number of examples
        Q = new double[Nc.length];
        for(int i = 0; i < Q.length; i++) {
            Q[i] = Nc[i] / training_set.getNumExamples();
        }
        
        //Initialize F
        //int max_bins = Arrays. IfsjdhflajhdfljksahfJKHDFKLJHDLJ
        int max_bins = 2;
        F = new double[training_set.getNumClasses()][training_set.getNumAttributes()][max_bins];
        //Calculate F for each attribute in/for each class
        //Start by counting examples that match
        for(int c = 0; c < training_set.getNumClasses(); c++) { //Iterate through the classes
            for(Example e : examples) { //Find all examples in that class
                if(e.getClassType() == c) {
                   for(int a = 0; a < training_set.getNumAttributes(); a++) { //Access all the example's attributes
                       F[c][a][e.getAttributes().get(a)]++; //Add one to the count of that attribute
                   }
                }
            }
        }
        //Finish calculating F by adding 1 and dividing by (#examples in the class + # attributes)
        for(int c = 0; c < training_set.getNumClasses(); c++) { //Iterate through the classes
            for(int a = 0; a < training_set.getNumAttributes(); a++) { //Iterate through the attributes
                for(int a_value = 0; a_value < training_set.getNumBins()[a]; a_value++) { //Iterate through the possible values of the attributes
                    F[c][a][a_value] = (F[c][a][a_value] + 1) / (Nc[c] + training_set.getNumAttributes());
                }
            }
        }
        
        //Training complete!
    }
    
    /**
     * Tests the algorithm with given testing set(s) and outputs the data used
     * for loss metrics.
     * @param test_set The test sets to test the algorithm with.
     */
    public void test(Set test_set) {
        ArrayList<Example> examples = test_set.getExamples();
        for(Example example : examples) { //Classify each example in the set
            // C = Q * the product of all relevant F
            ArrayList<Integer> attributes = example.getAttributes();
            double[] Cs = new double[num_classes];
            //Calculate C of each class
            for(int classid = 0; classid < num_classes; classid++) {
                int product = 1;
                //Find the product of all relevant F of each attribute
                for(int attrid = 0; attrid < num_attributes; attrid++) {
                    product *= F[classid][attrid][attributes.get(attrid)];
                }
                Cs[classid] = Q[classid] * product;
            }
            //Choose the class based on the maximum value of C in Cs.
            int classification = 0;
            for(int i = 1; i < Cs.length; i++) {
                if(Cs[i] > Cs[classification]){
                    classification = i;
                }
            }
            System.out.println("Class is: " + classification); //OUTPUT CLASSIFICATION
        }
    }
    
}
