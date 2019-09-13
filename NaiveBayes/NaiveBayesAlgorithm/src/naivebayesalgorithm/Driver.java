package naivebayesalgorithm;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;

/**
 * Class to perform the Naive Bayes algorithm
 * 
 * This class contains the main method and will
 * perform Naive Bayes on different data sets
 * and output the results into a file
 * 
 * @author natha
 */
public class Driver {
    
    public static final String OUTPUT_FILEPATH = "../output.csv";
    
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        
        // Make the list of data files we want to work with
        String[] datafiles = {"glass.csv", "glass-scrambled.csv",
            "iris.csv", "iris-scrambled.csv", 
            "house-votes-84.csv", "house-votes-84-scrambled.csv",
            "soybean-small.csv", "soybean-small-scrambled.csv",
            "wdbc.csv", "wdbc-scrambled.csv"};

        // Open the output file
        PrintWriter writer = new PrintWriter(OUTPUT_FILEPATH, "UTF-8");
        
        writer.println("Dataset,Accuracy,MSE");
        
        // Iterate through each data file
        for(int f = 0; f < datafiles.length; f++) {
            System.out.println("--- Handling " + datafiles[f] + " data set ---");
            //writer.println("--- Handling " + datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(datafiles[f]); 
            // Initialize the object that will be runnning our algorithm on the data
            NaiveBayes nb = new NaiveBayes();  
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            
            for(int i = 0; i < 10; i++) { // Perform 10-fold cross validation
                //System.out.println("Test " + (i+1));

                Set training_set = new Set(reader.getSubsets(), i); // Combine 9 of the subsets

                nb.train(training_set); // Train
                Set testing_set = reader.getSubsets()[i]; // Test with the remaining subset

                int[] predictions = nb.test(testing_set); // Test

                // Now, create the confusion matrix used to evaluate our loss metrics
                ConfusionMatrix m = new ConfusionMatrix(testing_set, predictions);
                double accuracy = m.getAccuracy();
                accuracy_sum += accuracy;
                double mse = m.getMSE();
                mse_sum += mse;
                // Output information about the metrics
                /* System.out.println("The accuracy was: " + 
                        new DecimalFormat("###.##").format(accuracy*100)
                        + "%");
                System.out.println("The MSE was: " + 
                        new DecimalFormat("###.##").format(mse));

                System.out.println(); */
            }
            // Output information about the loss metrics to the console
            System.out.println("Average accuracy for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            System.out.println("Average MSE for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
            System.out.println("----------------------------------------------");
            // Output information about the loss metrics to a file
            /*writer.println("Average accuracy for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            writer.println("Average MSE for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
            writer.println("----------------------------------------------");
            */
            
            writer.print(datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + "\n");
        }
        
        writer.close(); //Close output file
    }
    
}
