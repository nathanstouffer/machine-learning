package naivebayesalgorithm;

import java.util.ArrayList;

/**
 *
 * @author andy-
 */
public class ConfusionMatrix {
    
    private final int num_examples; // The number of examples used in the test set
                              // (Should the total of the confusion matrix).
    
    int[][] matrix; // Holds the confusion matrix, 
                    // indexed by matrix[<prediction>][<actual>]
    
    /**
     * 
     * @param test_set The set that was tested.
     * @param predictions The predictions of the classes for each example in
     * the set. Must be ordered by example in the set.
     */
    public ConfusionMatrix(Set test_set, int[] predictions) {
        num_examples = test_set.getNumExamples();
        
        // Initialize the matrix with all 0s
        matrix = new int[test_set.getNumClasses()][test_set.getNumClasses()];
        
        ArrayList<Example> examples = test_set.getExamples();
        // Populate the confusion matrix with totals
        for(int i = 0; i < examples.size(); i++) {
            matrix[predictions[i]][examples.get(i).getClassType()]++;
        }
    }
    
    /**
     * 
     * @return The accuracy of the classification according to the confusion
     * matrix. Calculated by taking the total number of correct classifications
     * divided by the total number of examples in the set.
     */
    public double getAccuracy() {
        int correct = 0;
        for(int i = 0; i < matrix.length; i++) {
            correct += matrix[i][i];
        }
        return (double) correct / (double) num_examples;
    }
    
    public double getClassPrecision(int classid) {
        return 0.0;
    }
    
    
    
}
