package nearestneighboralgorithm;

import java.util.ArrayList;

/**
 * Class to create and store a Confusion Matrix for analysis purposes
 * The object will store a confusion matrix and a client will be able
 * to access the accuracy and mean squared error for the sent into the
 * constructor
 * 
 * @author andy-
 */
public class ConfusionMatrix {
    
    private final int num_examples; // The number of examples used in the test set
                              // (Should the total of the confusion matrix).
    
    int[][] matrix; // Holds the confusion matrix, 
                    // indexed by matrix[<prediction>][<actual>]
    
    double mse; //Holds the mean squared error
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
        
        // populate confusion matrix with totals
        int i = 0;
        for (Example ex: test_set){ matrix[predictions[i++]][ex.getClassType()]++; }
        
        // outdated code since implementing Interable interface in Set class
        /*
        ArrayList<Example> examples = test_set.getExamples();
        // Populate the confusion matrix with totals
        for(int i = 0; i < examples.size(); i++) {
            matrix[predictions[i]][examples.get(i).getClassType()]++;
        }
        */
        
        //---------------------------------------------------------------
        //Calculate the mean squared error
        //Start by adding up the class predictions and the actual classes
        int[] actual_class_totals = new int[test_set.getNumClasses()];
        int[] pred_class_totals = new int[test_set.getNumClasses()];
        i = 0;
        for (Example ex: test_set){
            actual_class_totals[ex.getClassType()]++;
            pred_class_totals[predictions[i++]]++;
        }
        // outdated code since implementing Iterable interface in Set class
        /*
        for(int i = 0; i < examples.size(); i++) {
            actual_class_totals[examples.get(i).getClassType()]++;
            pred_class_totals[predictions[i]]++;
        }
        */
        //Find the difference
        double distances_sum = 0;
        i = 0;
        for (Example ex: test_set){
            distances_sum += Math.pow((pred_class_totals[i] - actual_class_totals[i]), 2);
            i++;
        }
        // outdated code since implementing Iterable interface in Set class
        /*
        for(int i = 0; i < actual_class_totals.length; i++) {
            distances_sum += Math.pow((pred_class_totals[i] - actual_class_totals[i]), 2);
        }
        */
        //Take the average
        mse = distances_sum / test_set.getNumClasses();
        
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
    
    /**
     * 
     * @return The Mean Squared Error, i.e., subtract the number of actual
     * examples in each class from the number predicted, square it, then add
     * the squares, taking the average.
     */ 
    public double getMSE() {
        return mse;
    }
    
    
    
}
