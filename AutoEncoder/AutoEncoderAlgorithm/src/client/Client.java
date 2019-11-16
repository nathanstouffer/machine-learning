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
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import neuralnets.*;

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
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        // read in data
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }
        
        String input = "none.csv";
        int INDEX = 0;
        int num_encoders = 1;
        double learning_rate = 0.1;
        double sparsity_penalty = 0.0005;
        double batch_size = 0.1;
        double momentum = 0.25;
        double convergence_threshold = 0.00000001;
        int max_iterations = 100000;
        int folds = 1;
        
        debugAE("test-ae.csv", datafiles[INDEX], data[INDEX], sparsity_penalty,
                learning_rate, momentum, batch_size, convergence_threshold, 
                max_iterations, folds);
        
        /*stackSAE("test-stacking.csv", datafiles[INDEX], data[INDEX], input, num_encoders,
                sparsity_penalty, learning_rate, momentum, batch_size,
                convergence_threshold, max_iterations, folds);
        */
    }
    
    private static void stackSAE(String output_file, String data_set, DataReader data, 
            String network_fname, int num_encoders, double sparsity_penalty, 
            double learning_rate, double momentum, double batch_size,
            double convergence_threshold, int max_iterations, 
            int folds) throws FileNotFoundException, UnsupportedEncodingException {
        
        System.out.println("STACKING NETWORKS ON DATASET " + data_set + " WITH " + num_encoders
                + " ENCODING LAYERS");
        System.out.println("LEARNING RATE: " + learning_rate + " WITH THRESH: " +
                convergence_threshold);
        System.out.println("MOMENTUM: " + momentum);
        
        SAE stacker = new SAE(num_encoders, sparsity_penalty, learning_rate, momentum, 
                                batch_size, convergence_threshold, max_iterations, 
                                data.getSimMatrices());
        
        // read in existing network at specified file name
        stacker.readExistingLayers(network_fname);
        
        double starttime = System.currentTimeMillis();

        // Initialize metrics
        double metric1 = 0;
        double metric2 = 0;

        // Perform 10-fold cross validation
        for(int i = 0; i < folds; i++) {
            System.out.println("Performing CV Fold #" + i);

            Set training_set = new Set(data.getSubsets(), i);
            Set testing_set = data.getSubsets()[i];

            stacker.train(training_set);
            
            // COMPUTE METRICS
        }

        // Take average of metrics
        metric1 /= folds;
        metric2 /= folds;
        
        
        // Create output string
        String output = data_set + "," + num_encoders + "," + sparsity_penalty + "," 
                + learning_rate + "," + momentum; // + "," + metric1 + "," + metric2;
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
            Set results = ae.testAutoEncoder(testing_set);
            
            // COMPUTE METRICS
            AutoEncoderEvaluator eval = new AutoEncoderEvaluator(results, testing_set);
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
}
