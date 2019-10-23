/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package client;

import datastorage.Set;
import knearestneighbor.*;
import evaluatelearner.*;
import reducedata.*;
import measuredistance.*;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;

/**
 *
 * @author natha
 */
public class Client {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        // List the files we want to test
        String[] knn_datafiles = {"abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv", "winequality-white.csv"};
        
        for (int f = 0; f < knn_datafiles.length; f++) {
            System.out.println("--- Handling " + knn_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(knn_datafiles[f]);
            
            // get the subsets
            Set[] subsets = reader.getSubsets();
            
            // determine whether data set is classification
            boolean classification = true;
            if (subsets[0].getClassNames() == null) { classification = false; }
            
            
        }
        
        testENN();
        testKNNCondensed();
    }
    
    private static void testENN() throws FileNotFoundException, UnsupportedEncodingException {
        // List the files we want to test
        String[] edited_datafiles = {"abalone.csv", "car.csv", "segmentation.csv"};

        // Iterate through each data file
        for(int f = 0; f < edited_datafiles.length; f++) {
            System.out.println("--- Handling " + edited_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(edited_datafiles[f]);

            // Initialize the object that will be runnning our algorithm on the data
            IKNearestNeighbor knn;
            knn = new KNNClassifier();
            knn.setDistMetric(new EuclideanSquared(reader.getSimMatrices()));

            //System.out.println("readervalidset" + reader.getValidationSet());
            //Edited edited_knn = new Edited((int)Math.sqrt(reader.getNumExamples()), new EuclideanSquared(reader.getSimMatrices()), reader.getValidationSet());
            Edited edited_knn = new Edited(1, new EuclideanSquared(reader.getSimMatrices()), reader.getValidationSet());


            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double num_points_sum = 0;

            // PERFORM 9 FOLD CROSS VALIDATION
            for(int i = 1; i < 10; i++) {
                System.out.println("Test " + (i));
                Set training_set = new Set(reader.getSubsets(), i, true); // Combine 9 of the subsets

                Set edited_set = edited_knn.reduce(training_set.clone());
                knn.setK((int)Math.sqrt(edited_set.getNumExamples()));

                System.out.println("EDITED SIZE: " + edited_set.getNumExamples());
                num_points_sum += edited_set.getNumExamples();

                knn.train(edited_set); // Train
                Set testing_set = reader.getSubsets()[i]; // Test with the remaining subset

                double[] predictions = knn.test(testing_set); // Test

                IEvaluator eval;
                eval = new ClassificationEvaluator(predictions, testing_set);

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
            // Output information about the loss metrics to the console----
            System.out.println("Average number of edited points for " + edited_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(num_points_sum/9));
            System.out.println("Average accuracy for " + edited_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(accuracy_sum/9*100)
                        + "%");
            System.out.println("Average MSE for " + edited_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(mse_sum/9));

            System.out.println("----------------------------------------------");

        }
    }
    
    
    private static void testKNNCondensed() throws FileNotFoundException, UnsupportedEncodingException {
        // List the files we want to test
        String[] condensed_datafiles = {"abalone.csv", "car.csv", "segmentation.csv"};

        // Iterate through each data file
        for(int f = 0; f < condensed_datafiles.length; f++) {
            System.out.println("--- Handling " + condensed_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(condensed_datafiles[f]);

            // Initialize the object that will be runnning our algorithm on the data
            IKNearestNeighbor knn;
            knn = new KNNClassifier();
            knn.setDistMetric(new EuclideanSquared(reader.getSimMatrices()));

            Condensed condensed_knn = new Condensed(new EuclideanSquared(reader.getSimMatrices()));

            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double num_points_sum = 0;

            // PERFORM 10 FOLD CROSS VALIDATION
            for(int i = 0; i < 10; i++) {
                System.out.println("Test " + (i+1));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subsets

                Set condensed_set = condensed_knn.reduce(training_set);

                knn.setK((int)Math.sqrt(condensed_set.getNumExamples()));

                System.out.println("ORIG SIZE: "+ training_set.getNumExamples());
                System.out.println("CONDENSED SIZE: " + condensed_set.getNumExamples());
                num_points_sum += condensed_set.getNumExamples();

                knn.train(condensed_set); // Train
                Set testing_set = reader.getSubsets()[i]; // Test with the remaining subset

                double[] predictions = knn.test(testing_set); // Test

                IEvaluator eval;
                eval = new ClassificationEvaluator(predictions, testing_set);

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
            // Output information about the loss metrics to the console----
            System.out.println("Average number of condensed points for" + condensed_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(num_points_sum/10));
            System.out.println("Average accuracy for " + condensed_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            System.out.println("Average MSE for " + condensed_datafiles[f] + " was "
                    + new DecimalFormat("###.##").format(mse_sum/10));

            System.out.println("----------------------------------------------");

        }
    }
}
