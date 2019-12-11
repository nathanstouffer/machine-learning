/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package client;

import datastorage.Set;
import evaluatelearner.AutoEncoderEvaluator;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import neuralnets.*;
import neuralnets.layer.Vector;

/**
 *
 * @author natha
 */
public class Client {
    
    private static String[] datafiles = {"abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv"}; //, "winequality-white.csv"};
    private static DataReader[] data = new DataReader[datafiles.length];

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        // read in data
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }
        
        // ------------------------------------------------------------
        // --- RUN FINAL AE TESTS WITH OPTIMAL PARAMETERS SELECTED ----
        // ------------------------------------------------------------
        //tuneAE();
        finalAE();

    }
    
    private static void finalAE() throws FileNotFoundException, IOException {
        System.out.println("--------- TUNING AE CONFIG ---------");

        String fout = "../Output/" + "AE-final-out.csv";
        clearFile(fout);
        
        double learning_rate = 0.1;
        double sparsity_penalty = 0.001;
        double batch_size = 0.1;
        double momentum = 0.25;
        double convergence_threshold = 0.00000001;
        int max_iterations = 5000;
        int folds = 10;
        
        // iterate through files
        for (int f = 0; f < datafiles.length; f++) {
            // iterate through encoders
            for (int num_encoders = 1; num_encoders < 4; num_encoders++) {
                String inputfname = datafiles[f].replace(".csv", "") + "-" + (num_encoders-1) + "-layer-ae.csv";
                runSAE(fout, datafiles[f], data[f], inputfname, 
                        num_encoders, sparsity_penalty, learning_rate, momentum, 
                        batch_size, convergence_threshold, max_iterations, folds);
            }
        }
    }
    
    private static void runSAE(String output_file, String data_set, DataReader data, 
            String network_fname, int num_encoders, double sparsity_penalty, 
            double learning_rate, double momentum, double batch_size,
            double convergence_threshold, int max_iterations, 
            int folds) throws FileNotFoundException, UnsupportedEncodingException, IOException {
        
        System.out.println("STACKING NETWORKS ON DATASET " + data_set + " WITH " + num_encoders
                + " ENCODING LAYERS");
        System.out.println("LEARNING RATE: " + learning_rate + " WITH THRESH: " +
                convergence_threshold);
        System.out.println("MOMENTUM: " + momentum);
        
        double starttime = System.currentTimeMillis();

        // Initialize metrics
        double metric1 = 0;
        double metric2 = 0;

        // Perform 10-fold cross validation
        for(int i = 0; i < folds; i++) {
            System.out.println("Performing CV Fold #" + i);

            Set training_set = new Set(data.getSubsets(), i);
            Set testing_set = data.getSubsets()[i];
            
            SAE stacker = new SAE(num_encoders, sparsity_penalty, learning_rate, momentum, 
                                batch_size, convergence_threshold, max_iterations, 
                                data.getSimMatrices());
            // read in existing network at specified file name
            if (num_encoders > 1) { stacker.readExistingLayers(network_fname); }
            
            stacker.train(training_set);
            
            double[] predicted = stacker.test(testing_set);
            
            // COMPUTE METRICS
            if (testing_set.getNumClasses() == -1) {
                // regression
                RegressionEvaluator eval = new RegressionEvaluator(predicted, testing_set);
                metric1 += eval.getMSE();
                metric2 += eval.getME();
            }
            else {
                // classification
                ClassificationEvaluator eval = new ClassificationEvaluator(predicted, testing_set);
                metric1 += eval.getAccuracy();
                metric2 += eval.getMSE();
            }
        }

        // Take average of metrics
        metric1 /= folds;
        metric2 /= folds;
        
        // Create output string
        String output = data_set + "," + num_encoders + "," + sparsity_penalty + "," 
                + learning_rate + "," + momentum + "," + metric1 + "," + metric2;
        // Write to file
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true)); // append = true
        writer.println(output);
        writer.close();
        
        // Write to console
        double endtime = System.currentTimeMillis();
        double runtime = (endtime - starttime) / 1000;
        System.out.println("\u001B[33m" + "Network trained and tested in " + runtime + " seconds");
        System.out.println("\u001B[33m" + output);
        
    }
    
    private static void debugAE(String output_file, String data_set, DataReader data,
            double sparsity_penalty, double learning_rate, double momentum,
            double batch_size, double convergence_threshold, int maximum_iterations,
            int folds) throws FileNotFoundException, UnsupportedEncodingException {
        
        System.out.println("TESTING AUTOENCODER ON DATASET " + data_set);
        System.out.println("LEARNING RATE: " + learning_rate + " WITH THRESH: " +
                convergence_threshold);
        System.out.println("MOMENTUM: " + momentum);
        
        double starttime = System.currentTimeMillis();

        // Initialize metrics
        double metric1 = 0;
        double metric2 = 0;

        // Perform 10-fold cross validation
        for(int i = 0; i < folds; i++) {
            System.out.println("Performing CV Fold #" + i);

            Set training_set = new Set(data.getSubsets(), i);
            Set testing_set = data.getSubsets()[i];

            AutoEncoder ae = new AutoEncoder(sparsity_penalty, learning_rate, 
                                        batch_size, momentum, convergence_threshold, 
                                        maximum_iterations, data.getSimMatrices());
            
            ae.train(training_set);
            //Set results = ae.testAutoEncoder(testing_set);
            Vector[] results = ae.testAutoEncoder(testing_set);
            Vector[] actual = new Vector[testing_set.getNumExamples()];
            for (int j = 0; j < actual.length; j++) {
                actual[j] = new Vector(testing_set.getExample(j), data.getSimMatrices());
            }
            
            // COMPUTE METRICS
            //AutoEncoderEvaluator eval = new AutoEncoderEvaluator(results, testing_set);
            AutoEncoderEvaluator eval = new AutoEncoderEvaluator(results, actual);
            metric1 += eval.getMSE();
            metric2 += eval.getMAE();
        }

        // Take average of metrics
        metric1 /= folds;
        metric2 /= folds;
        
        
        // Create output string
        String output = data_set + "," + sparsity_penalty + "," 
                + learning_rate + "," + momentum + "," + metric1 + "," + metric2;
        /*// Write to file
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true)); // append = true
        writer.println(output);
        writer.close();
        */
        // Write to console
        double endtime = System.currentTimeMillis();
        double runtime = (endtime - starttime) / 1000;
        System.out.println("\u001B[33m" + "Network trained and tested in " + runtime + " seconds");
        System.out.println("\u001B[33m" + output);
    }

    /**
     * method to clear file before writing
     * @param filename
     * @throws FileNotFoundException
     */
    private static void clearFile(String filename) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(filename);
        writer.print("");
        writer.close();
    }
}
