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
import measuredistance.EuclideanSquared;
import neuralnets.Clusterer;
import neuralnets.RBF;
import neuralnets.MLP;

/**
 *
 * @author natha
 */
public class Client {

    private static String[] datafiles = {"abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv"}; //, "winequality-white.csv"};
    private static DataReader[] data = new DataReader[datafiles.length];

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        // READ IN DATA
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }

        // ------------------------------------------------------------
        // --- TUNE MLP -----------------------------------------------
        // ------------------------------------------------------------

        // Tune learning rate
        //tuneMLPLearningRate();
        // Tune momentum
        //tuneMLPMomentum();
        // Tune number of hidden nodes
        //tuneMLPHiddenNodes();

        // ------------------------------------------------------------
        // --- TUNE RBF -----------------------------------------------
        // ------------------------------------------------------------

        // Tune learning rate
//        tuneRBF_learning_rate();

        // ------------------------------------------------------------
        // --- RUN FINAL MLP TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------

        finalMLP();

        // ------------------------------------------------------------
        // --- RUN FINAL RBF TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------
        //finalRBF();
        //demo();

    }
    private static void demo() 
            throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println(" --- DEMO ---");
        
        String output_file = "../Output/" + "demo-out.csv";
        clearFile(output_file);
        double[] rbf_learning_rates = {0.1, 0.5, 0.3, 0.001, 0.001, 0.01};
        //                 kmeans, pam, kmeans, kmeans, pam,  pam   
        int[] clust_meth = {1,      2,  1,      1,      2,     2};
        int k = 25;
        double conv_thresh = 0.0000000001;
        int rbf_max_iter = 100;
        double mlp_final_learning_rates = .1;
        double final_momentum = 0.5;
        double final_batch_size = 0.1;
        double final_hidden_nodes_mult = 2.0;         // multiply by the number of attributes to compute the number of hidden nodes
        int mlp_final_max_iterations = 10000;
        int folds = 1;
        //run on car data set
        System.out.println("MLP 1");
        DataReader curr_data = data[1];
        runMLP(output_file, datafiles[1], curr_data,
                        2, final_hidden_nodes_mult, mlp_final_learning_rates,
                        final_batch_size, final_momentum, conv_thresh,
                        mlp_final_max_iterations, folds);
        //run on machine data set
        System.out.println("MLP 2");
        curr_data = data[4];
        runMLP(output_file, datafiles[4], curr_data,
                        2, final_hidden_nodes_mult, mlp_final_learning_rates,
                        final_batch_size, final_momentum, conv_thresh,
                        mlp_final_max_iterations, folds);
        
        
        // Cluster data
        System.out.println("CLUSTERING DATA SETS");
        Clusterer[] clusters = new Clusterer[data.length];
        clusters[0] = new Clusterer(new EuclideanSquared(data[4].getSimMatrices()));
        clusters[0].cluster(data[4].getSubsets(), k);
        // output information
        System.out.println(String.format("num clusters for %s: %d", datafiles[4],
                clusters[0].getReps()[2].getNumExamples()));
        clusters[1] = new Clusterer(new EuclideanSquared(data[1].getSimMatrices()));
        clusters[1].cluster(data[1].getSubsets(), k);
        // output information
        System.out.println(String.format("num clusters for %s: %d", datafiles[1],
                clusters[1].getReps()[2].getNumExamples()));
        
        System.out.println("RBF 1");
        int dataset = 4;
        runRBF(output_file, datafiles[dataset], data[dataset],
                        k,
                        clusters[0].getReps(), clusters[0].getVars(),
                        rbf_learning_rates[dataset], final_batch_size,
                        conv_thresh, rbf_max_iter,
                        folds);
        System.out.println("RBF 2");
        dataset = 1;
        runRBF(output_file, datafiles[dataset], data[dataset],
                        k,
                        clusters[1].getReps(), clusters[1].getVars(),
                        rbf_learning_rates[dataset], final_batch_size,
                        conv_thresh, rbf_max_iter,
                        folds);
        
        
        
    }
    
    /**
     * method to run the final configuration of a MLP network
     * @param datafiles
     * @param data
     */
    private static void finalMLP()
            throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println(" --- TESTING FINAL MLP CONFIG ---");

        String output_file = "../Output/" + "MLP-final-out.csv";
        clearFile(output_file);
        // best learning rates for each dataset
        double[] final_learning_rates = { 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
        double final_momentum = 0.25;
        double final_batch_size = 0.1;
        double final_convergence_thresh =  0.0000000001;
        double final_hidden_nodes_mult = 2.0;         // multiply by the number of attributes to compute the number of hidden nodes
        int final_max_iterations = 100000;
        int final_num_folds = 10;

        // iterate through data files
        for (int f = 0; f < data.length; f++) {
            // get current dataset
            DataReader curr_data = data[f];
            // iterate through number of layers
            for (int num_layers = 0; num_layers < 3; num_layers++) {
                // run the network
                runMLP(output_file, datafiles[f], curr_data,
                        num_layers, final_hidden_nodes_mult, final_learning_rates[f],
                        final_batch_size, final_momentum, final_convergence_thresh,
                        final_max_iterations, final_num_folds);
            }
        }
    }

    
    public static void finalRBF() throws FileNotFoundException, UnsupportedEncodingException {
       System.out.println("~~~ ~~~ ~~~ RUNNING FINAL RBF TESTS ~~~ ~~~ ~~~");

        String output = "../Output/" + "RBF_final.csv";
        clearFile(output);
        double[] learning_rates = {0.1, 0.5, 0.3, 0.001, 0.001, 0.01};
        //                 kmeans, pam, kmeans, kmeans, pam,  pam   
        int[] clust_meth = {1,      2,  1,      1,      2,     2};
        int k = 25;
        double batch_size = 0.10;
        double conv_thresh = 0.0000000001;
        int max_iter = 10000;
        int folds = 10; 
        // Cluster data
        System.out.println("CLUSTERING DATA SETS");
        Clusterer[] clusters = new Clusterer[data.length];
        for(int i = 0; i < data.length; i++) {
                clusters[i] = new Clusterer(new EuclideanSquared(data[i].getSimMatrices()));
                clusters[i].cluster(data[i].getSubsets(), k);
                // output information
                System.out.println(String.format("num clusters for %s: %d", datafiles[i],
                        clusters[i].getReps()[2].getNumExamples()));
        }
        //Iterate through data sets
        for(int dataset = 0; dataset < data.length; dataset++) {
                runRBF(output, datafiles[dataset], data[dataset],
                        k,
                        clusters[dataset].getReps(), clusters[dataset].getVars(),
                        learning_rates[dataset], batch_size,
                        conv_thresh, max_iter,
                        folds);
        }
    }
    
    /**
     * Runs the RBF network at a variety of given learning rates, outputting
     * the results to console.
     */
    public static void tuneRBF_learning_rate() throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("~~~ ~~~ ~~~ TUNING RBF LEARNING RATE ~~~ ~~~ ~~~");

        String output = "../Output/" + "RBF_tune_learning_rate.csv";
        clearFile(output);
        double[] learning_rates = {0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2, 5};
        int k = 25;
        double batch_size = 0.10;
        double conv_thresh = 0.0000000001;
        int max_iter = 10000;
        int folds = 1;
        // Cluster data
        System.out.println("CLUSTERING DATA SETS");
        Clusterer[] clusters = new Clusterer[data.length];
        for(int i = 0; i < data.length; i++) {
                clusters[i] = new Clusterer(new EuclideanSquared(data[i].getSimMatrices()));
                clusters[i].cluster(data[i].getSubsets(), k);
                // output information
                System.out.println(String.format("num clusters for %s: %d", datafiles[i],
                        clusters[i].getReps()[2].getNumExamples()));
        }
        //Iterate through data sets
        for(int dataset = 0; dataset < data.length; dataset++) {
            //Iterate through learning rates
            for(int lr = 0; lr < learning_rates.length; lr++) {
                runRBF(output, datafiles[dataset], data[dataset],
                        25, // K
                        clusters[dataset].getReps(), clusters[dataset].getVars(),
                        learning_rates[lr], batch_size,
                        conv_thresh, max_iter,
                        folds);
            }
        }
    }

    /**
     * method to tune the variances through our k value
     * @param datafiles
     * @param data
     */
    private static void tuneMLPLearningRate() throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("~~~ ~~~ ~~~ TUNING MLP LEARNING RATE ~~~ ~~~ ~~~");

        String output_file = "../Output/" + "MLP-tuning-out.csv";
        clearFile(output_file);
        double[] learning_rates = { 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0 };
        double convergence_thresh =  0.0000000001;
        double momentum = 0.0;
        // multiply by the number of attributes to compute the number of hidden nodes
        double hidden_nodes_mult = 2.0;
        double batch_size = 0.1;
        int[] max_iterations = { 500000, 500000, 500000, 500000, 500000, 500000, 500000 };

        // iterate through data files
        for (int f = 0; f < data.length; f++) {
            // get current dataset
            DataReader curr_data = data[f];
            // iterate through layers
            for (int num_layers = 0; num_layers < 3; num_layers++) {
                // iterate through learning rates
                for (int lr = 0; lr < learning_rates.length; lr++) {
                    runMLP(output_file, datafiles[f], curr_data,
                            num_layers, hidden_nodes_mult, learning_rates[lr],
                            batch_size, momentum, convergence_thresh,
                            max_iterations[f], 1);
                }
            }
        }
    }

    /**
     * method to tune the number of hidden nodes in a network
     * @param datafiles
     * @param data
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private static void tuneMLPHiddenNodes() throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("~~~ ~~~ ~~~ TUNING MLP NUMBER OF HIDDEN NODES ~~~ ~~~ ~~~");

        String output_file = "../Output/" + "MLP-hidden-nodes-out.csv";
        clearFile(output_file);
        double[] learning_rates = { 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
        double convergence_thresh =  0.0000000001;
        double momentum = 0.0;
        // multiply by the number of attributes to compute the number of hidden nodes
        double[] hidden_nodes_mult = { 2.0/3.0, 1.0, 4.0/3.0, 5.0/3.0, 2.0, 7.0/3.0 };
        double batch_size = 0.1;
        int[] max_iterations = { 500000, 500000, 500000, 500000, 500000, 500000, 500000 };

        // iterate through data files
        for (int f = 1; f < data.length; f++) {
            // get current dataset
            DataReader curr_data = data[f];
            // iterate through number of layers
            for (int num_layers = 0; num_layers < 3; num_layers++) {
                // iterate through learning rates
                for (int hn = 0; hn < hidden_nodes_mult.length; hn++) {
                    runMLP(output_file, datafiles[f], curr_data,
                            num_layers, hidden_nodes_mult[hn], learning_rates[f],
                            batch_size, momentum, convergence_thresh,
                            max_iterations[f], 1);
                }
            }
        }
    }

    /**
     * method to tune momentum for a network
     * @param datafiles
     * @param data
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     */
    private static void tuneMLPMomentum() throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("~~~ ~~~ ~~~ TUNING MLP MOMENTUM ~~~ ~~~ ~~~");
        String output_file = "../Output/" + "MLP-momentum-out.csv";
        clearFile(output_file);
        double[] learning_rate = { 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
        double convergence_thresh =  0.0000000001;
        // multiply by the number of attributes to compute the number of hidden nodes
        double[] hidden_nodes_mult = { 2.0, 7.0/3.0, 2.0 };
        double[] momentum = { 0.0, 0.25, 0.5 };
        double batch_size = 0.1;
        int[] max_iterations = { 500000, 500000, 500000, 500000, 500000, 500000, 500000 };

        // iterate through data files
        for (int f = 0; f < data.length; f++) {
            // get current dataset
            DataReader curr_data = data[f];
            // iterate through layers
            for (int num_layers = 0; num_layers < 3; num_layers++) {
                // iterate through momentums
                for (int m = 0; m < momentum.length; m++) {
                    runMLP(output_file, datafiles[f], curr_data,
                            num_layers, hidden_nodes_mult[f], learning_rate[f],
                            batch_size, momentum[m], convergence_thresh,
                            max_iterations[f], 1);
                }
            }
        }
    }

    /**
     * Runs the RBF network given the parameters.
     * @param output_file
     * @param data_set
     * @param data
     * @param k
     * @param representatives
     * @param variances
     * @param learning_rate
     * @param batch_size
     * @param convergence_threshold
     * @param max_iterations
     * @param folds Number of cross validation folds to run (for official use, do 10!). Must be between 1 and 10.
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void runRBF(String output_file, String data_set, DataReader data,
            int k, Set[] representatives, double[][] variances,
            double learning_rate, double batch_size,
            double convergence_threshold, int max_iterations,
            int folds) throws FileNotFoundException, UnsupportedEncodingException {

        String[] clustering_methods = {"Edited/Condensed", "KMeans", "PAM"};
        for(int meth = 0; meth < 3; meth++) {
            if(representatives[meth] == null) {meth++;} //Skip editted/condensed if regression set

            System.out.println("TESTING RBF ON DATASET " + data_set + " WITH " + clustering_methods[meth] + " CLUSTERING");
            System.out.println("LEARNING RATE: " + learning_rate + " WITH THRESH: " +
                    convergence_threshold);

            double starttime = System.currentTimeMillis();

            EuclideanSquared dist = new EuclideanSquared(data.getSimMatrices());

            // Initialize metrics
            double metric1 = 0;
            double metric2 = 0;

            // Perform 10-fold cross validation
            for(int i = 0; i < folds; i++) {

                System.out.println("Performing CV Fold #" + i);

                Set training_set = new Set(data.getSubsets(), i);
                Set testing_set = data.getSubsets()[i];

                // Train and test the RBF network
                RBF rbf = new RBF(representatives[meth], variances[meth],
                                    learning_rate, batch_size,
                                    convergence_threshold, max_iterations,
                                    dist);
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
            metric1 /= folds;
            metric2 /= folds;

            // Create output string
            String output = data_set + "," + clustering_methods[meth] + "," + k + "," + learning_rate + "," + metric1 + "," + metric2;
            // Write to file
            PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true /* append = true */));
            writer.println(output);
            writer.close();
            // Write to console
            double endtime = System.currentTimeMillis();
            double runtime = (endtime - starttime) / 1000;
            System.out.println("\u001B[35m" + "RBF trained and tested in " + runtime + " seconds");
            System.out.println("\u001B[35m" + output);
        }
    }

    /**
     * Runs the MLP network given the parameters.
     * @param output_file
     * @param data_set
     * @param data
     * @param num_hidden_layers
     * @param hidden_nodes_mult
     * @param learning_rate
     * @param batch_size
     * @param momentum send in 0.0 if no momentum
     * @param convergence_threshold
     * @param max_iterations
     * @param folds Number of cross validation folds to run (for official use, do 10!). Must be between 1 and 10.
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    private static void runMLP(String output_file, String data_set, DataReader data,
            int num_hidden_layers, double hidden_nodes_mult,
            double learning_rate, double batch_size, double momentum,
            double convergence_threshold, int max_iterations,
            int folds) throws FileNotFoundException, UnsupportedEncodingException {

        System.out.println("TESTING MLP ON DATASET " + data_set + " WITH " + num_hidden_layers
                + " HIDDEN LAYERS AND (" + hidden_nodes_mult + " * NUM_ATTR) HIDDEN NODES");
        System.out.println("LEARNING RATE: " + learning_rate + " WITH THRESH: " +
                convergence_threshold);
        System.out.println("MOMENTUM: " + momentum);

        double starttime = System.currentTimeMillis();

        // Initialize metrics
        double metric1 = 0;
        double metric2 = 0;

        // compute number of hidden nodes
        int[] num_hidden_nodes = new int[num_hidden_layers];
        for (int i = 0; i < num_hidden_nodes.length; i++) {
            num_hidden_nodes[i] = (int)(hidden_nodes_mult * data.getSubsets()[0].getNumAttributes());
        }

        // Perform 10-fold cross validation
        for(int i = 0; i < folds; i++) {

            System.out.println("Performing CV Fold #" + i);

            Set training_set = new Set(data.getSubsets(), i);
            Set testing_set = data.getSubsets()[i];

            // Train and test the MLP network
            MLP mlp = new MLP(num_hidden_layers, num_hidden_nodes, learning_rate,
                                batch_size, momentum, convergence_threshold,
                                max_iterations, data.getSimMatrices());
            mlp.train(training_set);
            double[] results = mlp.test(testing_set);

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
        metric1 /= folds;
        metric2 /= folds;

        // Create output string
        String output = data_set + "," + num_hidden_layers + "," + hidden_nodes_mult + ","
                + learning_rate + "," + momentum + "," + metric1 + "," + metric2;
        // Write to file
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(output_file), true /* append = true */));
        writer.println(output);
        writer.close();
        // Write to console
        double endtime = System.currentTimeMillis();
        double runtime = (endtime - starttime) / 1000;
        System.out.println("\u001B[35m" + "MLP trained and tested in " + runtime + " seconds");
        System.out.println("\u001B[35m" + output);
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
