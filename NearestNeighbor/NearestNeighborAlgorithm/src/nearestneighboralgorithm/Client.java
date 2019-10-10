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
    
    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("-----------------------------------------");
        System.out.println("------------------- KNN -----------------");
        System.out.println("-----------------------------------------");
        testKNN();
        
        System.out.println("-----------------------------------------");
        System.out.println("------------------- EDITED --------------");
        System.out.println("-----------------------------------------");
        //testENN();
        
        System.out.println("-----------------------------------------");
        System.out.println("------------------- CONDENSED -----------");
        System.out.println("-----------------------------------------");
        //testKNNCondensed();
        
        System.out.println("-----------------------------------------");
        System.out.println("------------------- CMEANS --------------");
        System.out.println("-----------------------------------------");
        //testCMeans();
        
        System.out.println("-----------------------------------------");
        System.out.println("------------------- CMEDOIDS --------------");
        System.out.println("-----------------------------------------");
        //testMedoids();
    }
    
    private static void testKNN() throws FileNotFoundException, UnsupportedEncodingException {
        // Open the output file
        PrintWriter writer = new PrintWriter("../knn_output.csv", "UTF-8");
        writer.println("Dataset,Accuracy,MSE");
        
        // ---------------------------------------------------------------------
        // ------------------ TEST K-NN ----------------------------------------
        // ---------------------------------------------------------------------
        // List the files we want to test
        String[] knn_datafiles = {"test.csv", "abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv", "winequality-white.csv"};
        
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
            //knn.setK(1);
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double mae_sum = 0;
            double me_sum = 0;
            
            // PERFORM 10 FOLD CROSS VALIDATION
            for(int i = 0; i < 10; i++) {
                System.out.println("Test " + (i+1));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subset
                
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
                System.out.println("The MAE was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                System.out.println("The ME was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                // Track sums for averages
                accuracy_sum += eval.getAccuracy();
                mse_sum += eval.getMSE();
                mae_sum += eval.getMAE();
                me_sum += eval.getME();

                System.out.println();
            }
            // Output information about the loss metrics to the console
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                System.out.println("Average MSE for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("Average MAE for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mae_sum/10));
                System.out.println("Average ME for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(me_sum/10));
                System.out.println("----------------------------------------------");
            } else { // Otherwise, the file contained a classification set
                System.out.println("Average accuracy for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
                System.out.println("Average MSE for " + knn_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("----------------------------------------------");
            }
            
            
            // Output information about the loss metrics to a file
            /*writer.println("Average accuracy for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            writer.println("Average MSE for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
            writer.println("----------------------------------------------");
            */
            
            writer.print(knn_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + ",");
            writer.print(new DecimalFormat("###.##").format(mae_sum/10) + ",");
            writer.print(new DecimalFormat("###.##").format(me_sum/10) + "\n");
        }          
        
        writer.close(); //Close output file
    }
    
    private static void testENN() throws FileNotFoundException, UnsupportedEncodingException {
        // Open the output file
        PrintWriter writer = new PrintWriter("../knn_edited_output.csv", "UTF-8");
        writer.println("Dataset,Accuracy,MSE,Datapoints");
        
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
           
            
            System.out.println("readervalidset" + reader.getValidationSet());
            //Edited edited_knn = new Edited((int)Math.sqrt(reader.getNumExamples()), new EuclideanSquared(reader.getSimMatrices()), reader.getValidationSet());      
            Edited edited_knn = new Edited(3, new EuclideanSquared(reader.getSimMatrices()), reader.getValidationSet());    
            
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double num_points_sum = 0;
            
            // PERFORM 9 FOLD CROSS VALIDATION
            for(int i = 1; i < 10; i++) {
                System.out.println("Test " + (i));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subsets

                Set edited_set = edited_knn.reduce(training_set.clone());
                knn.setK((int)Math.sqrt(edited_set.getNumExamples()));
                
                System.out.println("NUM POINTS EDITED: " + edited_set.getNumExamples());
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
                System.out.println("The MAE was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                System.out.println("The ME was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                // Track sums for averages
                accuracy_sum += eval.getAccuracy();
                mse_sum += eval.getMSE();

                System.out.println();
            }
            // Output information about the loss metrics to the console----
            System.out.println("Average number of condensed points for" + edited_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(num_points_sum/9));
            System.out.println("Average accuracy for " + edited_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/9*100)
                        + "%");
            System.out.println("Average MSE for " + edited_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/9));
            
            System.out.println("----------------------------------------------");
            
            
            writer.print(edited_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/9*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/9) + ",");
            writer.print(new DecimalFormat("###.##").format((double)num_points_sum/(double)9) + "\n");
            
        }          
        
        writer.close(); //Close output file
    }
    
    private static void testKNNCondensed() throws FileNotFoundException, UnsupportedEncodingException {
        // Open the output file
        PrintWriter writer = new PrintWriter("../knn_condensed_output.csv", "UTF-8");
        writer.println("Dataset,Accuracy,MSE,Datapoints");
        
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
            knn.setK((int)Math.sqrt(reader.getNumExamples()));
            
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
                
                System.out.println("Num points original: "+ training_set.getNumExamples());
                System.out.println("NUM POINTS CONDENSED: " + condensed_set.getNumExamples());
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
                System.out.println("The MAE was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                System.out.println("The ME was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
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
            
            
            writer.print(condensed_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + ",");
            writer.print(new DecimalFormat("###.##").format((double)num_points_sum/(double)10) + "\n");
            
        }          
        
        writer.close(); //Close output file
    }
    
    private static void testCMeans() throws FileNotFoundException, UnsupportedEncodingException {
        // Open the output file
        PrintWriter writer = new PrintWriter("../cmeans_output.csv", "UTF-8");
        writer.println("Dataset,Accuracy,MSE,Datapoints");
        
        // List the files we want to test
        String[] cmeans_datafiles = {"abalone.csv", "car.csv", "segmentation.csv", };
        
        // Iterate through each data file
        for(int f = 0; f < cmeans_datafiles.length; f++) {
            System.out.println("--- Handling " + cmeans_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(cmeans_datafiles[f]); 
            
            // Initialize the object that will be runnning our algorithm on the data
            IKNearestNeighbor knn;
            CMeans cmeans;
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                knn = new KNNRegressor();
                cmeans = new CMeans((int) 0.25 * reader.getNumExamples(), new EuclideanSquared(reader.getSimMatrices())); // Set clusters by 0.25 n
            } else { // Otherwise, the file contained a classification set
                knn = new KNNClassifier();
                cmeans = new CMeans(10, new EuclideanSquared(reader.getSimMatrices())); // Set clusters manually to result of E-NN
            }      
            knn.setDistMetric(new EuclideanSquared(reader.getSimMatrices()));
            knn.setK((int)Math.sqrt(reader.getNumExamples()));
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double mae_sum = 0;
            double me_sum = 0;
            
            // PERFORM 10 FOLD CROSS VALIDATION
            for(int i = 0; i < 10; i++) {
                System.out.println("Test " + (i+1));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subsets

                Set means_set = cmeans.reduce(training_set.clone());
                
                System.out.println("CMEANS REDUCED TO " + means_set.getNumExamples() + " POINTS");
                System.out.println(means_set.getExamples().toString());
                
                knn.train(means_set); // Train
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
                System.out.println("The MAE was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                System.out.println("The ME was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                // Track sums for averages
                accuracy_sum += eval.getAccuracy();
                mse_sum += eval.getMSE();
                mae_sum += eval.getMAE();
                me_sum += eval.getME();

                System.out.println();
            }
            // Output information about the loss metrics to the console----
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                System.out.println("Average MSE for " + cmeans_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("Average MAE for " + cmeans_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mae_sum/10));
                System.out.println("Average ME for " + cmeans_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(me_sum/10));
                System.out.println("----------------------------------------------");
            } else { // Otherwise, the file contained a classification set
                System.out.println("Average accuracy for " + cmeans_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
                System.out.println("Average MSE for " + cmeans_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("----------------------------------------------");
            }
            
            
            
            // Output information about the loss metrics to a file
            /*writer.println("Average accuracy for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            writer.println("Average MSE for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
            writer.println("----------------------------------------------");
            */
            
            writer.print(cmeans_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + "\n");
            
        }          
        
        writer.close(); //Close output file
    }
    
     private static void testMedoids() throws FileNotFoundException, UnsupportedEncodingException {
        // Open the output file
        PrintWriter writer = new PrintWriter("../medoids_output.csv", "UTF-8");
        writer.println("Dataset,Accuracy,MSE,Datapoints");
        
        // List the files we want to test
        String[] medoids_datafiles = {"car.csv"};//{"abalone.csv", "car.csv", "segmentation.csv", };
        
        // Iterate through each data file
        for(int f = 0; f < medoids_datafiles.length; f++) {
            System.out.println("--- Handling " + medoids_datafiles[f] + " data set ---");
            // Create the data reader to read in our preprocessed files
            DataReader reader = new DataReader(medoids_datafiles[f]); 
            
            // Initialize the object that will be runnning our algorithm on the data
            IKNearestNeighbor knn;
            CMedoids medoids;
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                knn = new KNNRegressor();
                medoids = new CMedoids((int) 0.25 * reader.getNumExamples(), new EuclideanSquared(reader.getSimMatrices())); // Set clusters by 0.25 n
            } else { // Otherwise, the file contained a classification set
                knn = new KNNClassifier();
                medoids = new CMedoids(5, new EuclideanSquared(reader.getSimMatrices())); // Set clusters manually to result of E-NN
            }      
            knn.setDistMetric(new EuclideanSquared(reader.getSimMatrices()));
            knn.setK((int)Math.sqrt(reader.getNumExamples()));
            
            // Initialize the sums that will be used to compute our average loss metrics
            double accuracy_sum = 0;
            double mse_sum = 0;
            double mae_sum = 0;
            double me_sum = 0;
            
            // PERFORM 10 FOLD CROSS VALIDATION
            for(int i = 0; i < 10; i++) {
                System.out.println("Test " + (i+1));
                Set training_set = new Set(reader.getSubsets(), i, false); // Combine 9 of the subsets

                Set medoids_set = medoids.reduce(training_set.clone());
                
                System.out.println("MEDOIDS REDUCED TO " + medoids_set.getNumExamples() + " POINTS");
                System.out.println(medoids_set.getExamples().toString());
                
                knn.train(medoids_set); // Train
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
                System.out.println("The MAE was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                System.out.println("The ME was: " + 
                        new DecimalFormat("###.##").format(eval.getMAE()));
                // Track sums for averages
                accuracy_sum += eval.getAccuracy();
                mse_sum += eval.getMSE();
                mae_sum += eval.getMAE();
                me_sum += eval.getME();

                System.out.println();
            }
            // Output information about the loss metrics to the console----
            if(reader.getClassNames() == null) { // Check if the file contained a regression set
                System.out.println("Average MSE for " + medoids_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("Average MAE for " + medoids_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mae_sum/10));
                System.out.println("Average ME for " + medoids_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(me_sum/10));
                System.out.println("----------------------------------------------");
            } else { // Otherwise, the file contained a classification set
                System.out.println("Average accuracy for " + medoids_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
                System.out.println("Average MSE for " + medoids_datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
                System.out.println("----------------------------------------------");
            }
            
            
            
            // Output information about the loss metrics to a file
            /*writer.println("Average accuracy for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(accuracy_sum/10*100)
                        + "%");
            writer.println("Average MSE for " + datafiles[f] + " was " 
                    + new DecimalFormat("###.##").format(mse_sum/10));
            writer.println("----------------------------------------------");
            */
            
            writer.print(medoids_datafiles[f] + "," + new DecimalFormat("###.##").format(accuracy_sum/10*100)+ "%,");
            writer.print(new DecimalFormat("###.##").format(mse_sum/10) + "\n");
            
        }          
        
        writer.close(); //Close output file
    }
    
}
