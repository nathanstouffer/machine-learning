/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package client;

import datastorage.Set;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
import measuredistance.EuclideanSquared;
import neuralnets.Clusterer;
import neuralnets.RBF;
import reducedata.IDataReducer;

/**
 *
 * @author natha
 */
public class Client {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        
        String[] datafiles = {"abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv", "winequality-white.csv"};
        
        // READ IN DATA
        DataReader[] data = new DataReader[datafiles.length];
        for(int i = 0; i < data.length; i++) {
            data[i] = new DataReader(datafiles[i]);
        }
        
        String output_file = "out.csv";
        
        // ------------------------------------------------------------
        // --- TUNE RBF -----------------------------------------------
        // ------------------------------------------------------------
        
        // Tune K
        // Tune learning rate
        
        // ------------------------------------------------------------
        // --- RUN FINAL RBF TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------
       
        System.out.println(" --- TESTING FINAL RBF CONFIG ---");
        
        output_file = "../Output/" + "RBF_out.csv";
        // << Add funct to clear file contents here >>
        int final_k = 20;
        double final_learning_rate = 0.001;
        double final_batch_size = 0.1;
        
        // CLUSTER DATA
        System.out.println("Clustering datasets...");
        Clusterer[] clusters = new Clusterer[datafiles.length];
        
        int TODO = 0;
        
        for(int i = 0; i < data.length; i++) {
            if(i == TODO) {
                clusters[i] = new Clusterer(new EuclideanSquared(data[i].getSimMatrices()));
                clusters[i].cluster(data[i].getSubsets(), final_k);
                // output information
                System.out.println(String.format("num clusters for %s: %d", datafiles[i], 
                        clusters[i].getReps()[2].getNumExamples()));
            }
        }

        // RUN TEST
//        runRBF(output_file, datafiles[TODO], data[TODO], final_k, clusters[TODO].getReps(), clusters[TODO].getVars(), final_learning_rate, final_batch_size);
        runRBFquick(output_file, datafiles[TODO], data[TODO], final_k, clusters[TODO].getReps(), clusters[TODO].getVars(), final_learning_rate, final_batch_size);
        
        
    }
    
    /**
     * Runs the RBF network given the parameters. Will run 10-fold cross validation
     * on the data set given to it.
     * @param output_file
     * @param data_set
     * @param data
     * @param k
     * @param representatives
     * @param variances
     * @param learning_rate
     * @param batch_size 
     * @throws java.io.FileNotFoundException 
     * @throws java.io.UnsupportedEncodingException 
     */
    public static void runRBF(String output_file, String data_set, DataReader data, 
            int k, Set[] representatives, double[][] variances, 
            double learning_rate, double batch_size) throws FileNotFoundException, UnsupportedEncodingException {
        
        String[] clustering_methods = {"Edited/Condensed", "KMeans", "PAM"};
        for(int meth = 0; meth < 3; meth++) {
            if(representatives[meth] == null) {meth++;} //Skip editted/condensed if regression set
            
            System.out.println("TESTING RBF ON DATASET " + data_set + " WITH " + clustering_methods[meth] + " CLUSTERING");
            
            double starttime = System.currentTimeMillis();

            EuclideanSquared dist = new EuclideanSquared(data.getSimMatrices());

            // Initialize metrics
            double metric1 = 0;
            double metric2 = 0;

            // Perform 10-fold cross validation
            for(int i = 0; i < 10; i++) {

                System.out.println("Performing CV Fold #" + i);

                Set training_set = new Set(data.getSubsets(), i);
                Set testing_set = data.getSubsets()[i];

                // Train and test the RBF network
                RBF rbf = new RBF(representatives[meth], variances[meth], learning_rate, batch_size, dist);
                rbf.train(training_set);
                double[] results = rbf.test(testing_set);

                // Get metrics
                if(training_set.getNumClasses() == -1) {
                    // Regression
                    RegressionEvaluator eval = new RegressionEvaluator(results, testing_set);
                    metric1 += eval.getMSE();
                    metric2 += eval.getME();
                } else {
                    // Classification
                    ClassificationEvaluator eval = new ClassificationEvaluator(results, testing_set);
                    metric1 += eval.getAccuracy();
                    metric2 += eval.getMSE();
                }


            }

            // Take average of metrics
            metric1 /= 10;
            metric2 /= 10;

            // Create output string
            String output = data_set + "," + clustering_methods[meth] + "," + k + "," + learning_rate + "," + metric1 + "," + metric2;
            // Write to file
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true /* append = true */));
            writer.println(output);
            writer.close();
            // Write to console
            double endtime = System.currentTimeMillis();
            double runtime = (endtime - starttime) / 1000;
            System.out.println("RBF trained and tested in " + runtime + " seconds");
            System.out.println(output);
        }
    }
    
    /**
     * Runs the RBF network given the parameters quickly (no 10 fold CV)
     * on the data set given to it.
     * @param output_file
     * @param data_set
     * @param data
     * @param k
     * @param representatives
     * @param variances
     * @param learning_rate
     * @param batch_size 
     * @throws java.io.FileNotFoundException 
     * @throws java.io.UnsupportedEncodingException 
     */
    public static void runRBFquick(String output_file, String data_set, DataReader data, 
            int k, Set[] representatives, double[][] variances, 
            double learning_rate, double batch_size) throws FileNotFoundException, UnsupportedEncodingException {
        
        String[] clustering_methods = {"Edited/Condensed", "KMeans", "PAM"};
        for(int meth = 0; meth < 3; meth++) {
            if(representatives[meth] == null) {meth++;} //Skip editted/condensed if regression set
            
            System.out.println("TESTING RBF ON DATASET " + data_set + " WITH " + clustering_methods[meth] + " CLUSTERING");
            
            double starttime = System.currentTimeMillis();

            EuclideanSquared dist = new EuclideanSquared(data.getSimMatrices());

            // Initialize metrics
            double metric1 = 0;
            double metric2 = 0;

            Random rand = new Random();
            int test_set_index = rand.nextInt(10);
            // Just use subset 1 for testing for quicks
            Set training_set = new Set(data.getSubsets(), test_set_index);
            Set testing_set = data.getSubsets()[test_set_index];

            System.out.println("CLUSTER VARIS: " + Arrays.toString(variances[meth]));
            
            // Train and test the RBF network
            RBF rbf = new RBF(representatives[meth], variances[meth], learning_rate, batch_size, dist);
            rbf.train(training_set);
            double[] results = rbf.test(testing_set);
            
            System.out.print("ORIGINAL: [");
            for(int i = 0; i < testing_set.getNumExamples(); i++) {System.out.print(testing_set.getExample(i).getValue() + ", ");}
            System.out.println("]");
            System.out.println("RESULTS: " + Arrays.toString(results));

            // Get metrics
            if(training_set.getNumClasses() == -1) {
                // Regression
                RegressionEvaluator eval = new RegressionEvaluator(results, testing_set);
                metric1 += eval.getMSE();
                metric2 += eval.getME();
            } else {
                // Classification
                ClassificationEvaluator eval = new ClassificationEvaluator(results, testing_set);
                metric1 += eval.getAccuracy();
                metric2 += eval.getMSE();
            }
            // Create output string
            String output = data_set + "," + clustering_methods[meth] + "," + k + "," + learning_rate + "," + metric1 + "," + metric2;
            // Write to file
//            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true /* append = true */));
//            writer.println(output);
//            writer.close();
            // Write to console
            double endtime = System.currentTimeMillis();
            double runtime = (endtime - starttime) / 1000;
            System.out.println("RBF trained and tested in " + runtime + " seconds");
            System.out.println("\u001B[35m" + output);
            System.out.println();
        }
    }
}
