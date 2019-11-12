package evaluatelearner;

import datastorage.Set;

/**
 *
 * @author andy-
 */
public class ClassificationEvaluator implements IEvaluator {
    
    double accuracy; //Holds the accuracy
    
    double mse; //Holds the mean squared error
    
    public ClassificationEvaluator(double[] predicted, Set actual) {
        //this.printPred(predicted);
        int num_examples = actual.getNumExamples();
        
        //---------------------------------------------------------------
        // Calculate accuracy 
        
        // Find number of items classified correctly
        accuracy = 0;
        for (int i = 0; i < actual.getNumExamples(); i++) {
            if (Math.round(actual.getExample(i).getValue()) == Math.round(predicted[i])) {
                accuracy++;
            }
        }
        // Divide by the number of examples to yield accuracy
        accuracy /= (double)num_examples;

        
        //---------------------------------------------------------------
        //Calculate the mean squared error
        //Start by adding up the class predictions and the actual classes
        int[] actual_class_totals = new int[actual.getNumClasses()];
        int[] pred_class_totals = new int[actual.getNumClasses()];
        for (int i = 0; i < actual.getNumExamples(); i ++) {
            actual_class_totals[(int)actual.getExample(i).getValue()]++;
            pred_class_totals[(int)predicted[i]]++;
        }
        //Find the difference
        double distances_sum = 0;
        for(int i = 0; i < actual_class_totals.length; i++) {
            distances_sum += Math.pow((pred_class_totals[i] - actual_class_totals[i]), 2);
        }
        //Take the average
        mse = distances_sum / actual.getNumClasses();
    }
    
    @Override
    public double getAccuracy() {
        return accuracy;
    }

    @Override
    public double getMSE() {
        return mse;
    }

    @Override
    public double getMAE() {
        return -1; // Unimplemented
    }

    @Override
    public double getME() {
        return -1; // Unimplemented
    }
    
    private void printPred(double[] pred) {
        String output = "PREDICTIONS:\n[";
        int line_count = 0;
        for (int i = 0; i < pred.length; i++) {
            line_count++;
            output += Double.toString(pred[i]) + ", ";
            if (line_count == 22) { 
                output += "\n "; 
                line_count = 0;
            }
        }
        output += pred[pred.length-1] + "]";
        System.out.println(output);
    }
    
}
