package evaluatelearner;

import datastorage.Set;

/**
 *
 * @author andy-
 */
public class RegressionEvaluator implements IEvaluator{
    
    double mse; //Holds the mean squared error
    
    double mae; //Holds the mean absolute error
    
    double me; //Holds the mean error
    
    public RegressionEvaluator(double[] predicted, Set actual) {
        this.printPred(predicted);
        int num_examples = actual.getNumExamples();
        
        // standardize the values using z-scores
        // the mean and standard deviation will be computed
        // using the values from the actual dataset
        
        // compute mean of training data
        double mean = 0.0;
        for (int i = 0; i < num_examples; i++) { mean += actual.getExample(i).getValue(); }
        mean /= num_examples;
        
        // compute standard deviation of training data
        double sd = 0.0;
        for (int i = 0; i < num_examples; i++) { 
            sd += Math.pow(actual.getExample(i).getValue() - mean, 2);
        }
        sd /= (num_examples - 1);
        sd = Math.sqrt(sd);
        
        // instantiate z-score arrays for predictions and actuals
        double[] actz = new double[num_examples];
        double[] predz = new double[num_examples];
        
        // populate z-score arrays
        for (int i = 0; i < num_examples; i++) {
            // find difference between value and mean
            actz[i] = actual.getExample(i).getValue() - mean;
            predz[i] = predicted[i] - mean;
            // divide by standard deviation
            actz[i] /= sd;
            predz[i] /= sd;
        }
        
        // the output has now been standardized using z-scores so
        // the metrics are comparable between datasets
        
        mse = 0;
        mae = 0;
        me = 0;
        //---------------------------------------------------------------
        // Calculate the mean squared error and mean absolute error
        for (int i = 0; i < actual.getNumExamples(); i++) {
            double difference = predz[i] - actz[i];
            me += difference;
            mae += Math.abs(difference);
            mse += Math.pow(difference, 2);
        }
        //Take the average
        mse /= num_examples;
        mae /= num_examples;
        me /= num_examples;
    }
    
    @Override
    public double getAccuracy() {
        return -1; // Unimplemented
    }

    @Override
    public double getMSE() {
        return mse;
    }

    @Override
    public double getMAE() {
        return mae;
    }
    
    @Override
    public double getME() {
        return me;
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
