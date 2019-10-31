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
import measuredistance.EuclideanSquared;
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
        
        DataReader[] data = new DataReader[datafiles.length];
        for(int i = 0; i < data.length; i++) {
            data[i] = new DataReader(datafiles[i]);
        }
        
        String output_file = "out.csv";
        
        // CREATE SOME CLASS TO RUN CLUSTERING, RETURNING 3 SETS OF CLUSTERS
        // (REPRESENTATIVES AND VARIANCES)
        // THEN WE CAN PASS THESE INTO RUN RBF INDIVIDUALLY 
        // Avoids this below    |
        //                      V
        //IDataReducer[] clusterers = new IDataReducer[3];
        //clusterers[0] = new Edited(); / new Condensed();
        //clusterers[1] = new CMeans();
        //clusterers[1] = new CMediods();
        
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
        int final_k = 10;
        double final_learning_rate = 0.50;
        double final_batch_size = 0.10;
        
        // Loop through data sets
//        for(int d = 0; d < datafiles.length; d++) {
//            // Loop through clustering methods
//            // Cluster data w/ object (passing in final k)
//            // Put each cluster through
//            for(int c = 0; c < 3; c++) {
//                runRBF(output_file, datafiles[d], data[d], 
//                        "clustermethod", final_k, null, null, 
//                        final_learning_rate, final_batch_size);
//            }
//        }

        int TODO = 1;
        Set rep = new Set(data[TODO].getSubsets()[0].getNumAttributes(), data[TODO].getSubsets()[0].getNumClasses(), data[TODO].getSubsets()[0].getClassNames());
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(0));
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(1));
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(2));
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(3));
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(4));
        rep.addExample(data[TODO].getSubsets()[0].getExamples().get(5));
        double[] var = new double[rep.getNumExamples()];
        for(int i = 0; i < var.length; i++) {
            var[i] = 10;
        }
        runRBF(output_file, datafiles[TODO], data[TODO], "clustermeth", final_k, rep, var, final_learning_rate, final_batch_size);
        
        
    }
    
    /**
     * Runs the RBF network given the parameters. Will run 10-fold cross validation
     * on the data set given to it.
     * @param output_file
     * @param data_set
     * @param data
     * @param clustering_method
     * @param representatives
     * @param variances
     * @param learning_rate
     * @param batch_size 
     * @throws java.io.FileNotFoundException 
     * @throws java.io.UnsupportedEncodingException 
     */
    public static void runRBF(String output_file, String data_set, DataReader data, 
            String clustering_method, int k, Set representatives, double[] variances, 
            double learning_rate, double batch_size) throws FileNotFoundException, UnsupportedEncodingException {
        
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
            RBF rbf = new RBF(representatives, variances, learning_rate, batch_size, dist);
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
        String output = data_set + "," + clustering_method + "," + k + "," + learning_rate + "," + metric1 + "," + metric2;
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
