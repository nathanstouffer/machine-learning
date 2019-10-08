package nearestneighboralgorithm;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;

/**
 *
 * @author andy-
 */
public class Client {

    public static final String OUTPUT_FILEPATH = "../output.csv";
    
    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        

        // Open the output file
        PrintWriter writer = new PrintWriter(OUTPUT_FILEPATH, "UTF-8");
        writer.println("Dataset,Accuracy,MSE");
        
        // ---------------------------------------------------------------------
        // ------------------ TEST K-NN ----------------------------------------
        // ---------------------------------------------------------------------
        // List the files we want to test
        String[] knn_datafiles = {"abalone.csv", "car.csv", "forestfires.csv", "machine.csv", "segmentation.csv", "winequality-red.csv", "winequality-white.csv"};
        
        // Iterate through each data file
        for(int f = 0; f < knn_datafiles.length; f++) {
            System.out.println("--- Handling " + knn_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(knn_datafiles[f]); 
            
            // Initialize the object that will be runnning our algorithm on the data
            IKNearestNeighbor knn;
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                knn = new KNNRegressor();
            } else { // Otherwise, the file contained a classification set
                knn = new KNNClassifier();
            }
            knn.setDistMetric(new EuclideanSquared(reader.getSimMatrices()));
            knn.setK((int)Math.sqrt(reader.getNumExamples()));
            System.out.println((int)Math.sqrt(reader.getNumExamples()));
            //knn.setK(5);
            
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            
            // PERFORM 10 FOLD CROSS VALIDATION
            for(int i = 0; i < 10; i++) {
                System.out.println("Test " + (i+1));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subsets

                knn.train(training_set); // Train
                Set testing_set = reader.getSubsets()[i]; // Test with the remaining subset

                double[] predictions = knn.test(testing_set); // Test
                
                IEvaluator eval;
                if(reader.getClassNames() == null) { // Check if the file contained a regression set
                    eval = new RegressionEvaluator(predictions, testing_set);
                } else { // Otherwise, the file contained a classification set
                    eval = new ClassificationEvaluator(predictions, testing_set);
                }
                
                // Output information about the metrics
                System.out.println("The accuracy was: " + 
                        new DecimalFormat("###.##").format(eval.getAccuracy()*100)
                        + "%");
                System.out.println("The MSE was: " + 
                        new DecimalFormat("###.##").format(eval.getMSE()));
                // Track sums for averages
                accuracy_sum += eval.getAccuracy();
                mse_sum += eval.getMSE();

                System.out.println();
            }
            // Output information about the loss metrics to the console
            System.out.println("Average accuracy for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            System.out.println("Average MSE for " + knn_datafiles[f] + " was " 
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
            
            writer.print(knn_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + "\n");
            
        }   
        // ---------------------------------------------------------------------
        // ------------------ TEST K-CONDENSED ----------------------------------------
        // ---------------------------------------------------------------------
        String[] condensed_datafiles = {"abalone.csv", "car.csv", "forestfires.csv"};
        
        
        writer.close(); //Close output file
    }
    
}
