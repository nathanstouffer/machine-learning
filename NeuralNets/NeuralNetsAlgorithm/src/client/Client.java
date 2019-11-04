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

    public static String[] datafiles = {"abalone.csv", "car.csv", "segmentation.csv", "forestfires.csv", "machine.csv", "winequality-red.csv", "winequality-white.csv"};
    public static DataReader[] data;

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.UnsupportedEncodingException
     */
    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        // READ IN DATA
        DataReader[] data = new DataReader[datafiles.length];
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }


        String output_file = "out.csv";

        // ------------------------------------------------------------
        // --- TUNE MLP -----------------------------------------------
        // ------------------------------------------------------------

        // Tune learning rate
        // Tune number of hidden nodes
        // Tune momentum
        tuneMLP(datafiles, data);

        // ------------------------------------------------------------
        // --- TUNE RBF -----------------------------------------------
        // ------------------------------------------------------------

        // Tune K
        // Tune learning rate
        System.out.println("~~~ ~~~ ~~~ TUNING RBF LEARNING RATE ~~~ ~~~ ~~~");
        tuneRBF_learning_rate();

        // ------------------------------------------------------------
        // --- RUN FINAL MLP TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------

        //finalMLP(datafiles, data);

        // ------------------------------------------------------------
        // --- RUN FINAL RBF TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------

//        System.out.println(" --- TESTING FINAL RBF CONFIG ---");

//        output_file = "../Output/" + "RBF_out.csv";
//        // << Add funct to clear file contents here >>
//        int final_k = 20;
//        double final_learning_rate = 0.001;
//        double final_batch_size = 0.1;
//        double final_convergence_thresh = 0.0001;
//        int final_max_iterations = 100000;
//
//
//        // CLUSTER DATA
//        System.out.println("Clustering datasets...");
//        Clusterer[] clusters = new Clusterer[datafiles.length];
//
//        int TODO = 1;
//
//        for(int i = 0; i < data.length; i++) {
//            if(i == TODO) {
//                clusters[i] = new Clusterer(new EuclideanSquared(data[i].getSimMatrices()));
//                clusters[i].cluster(data[i].getSubsets(), final_k);
//                // output information
//                System.out.println(String.format("num clusters for %s: %d", datafiles[i],
//                        clusters[i].getReps()[2].getNumExamples()));
//            }
//        }
//
//        // RUN TEST
//        runRBF(output_file, datafiles[TODO], data[TODO],
//                final_k, clusters[TODO].getReps(), clusters[TODO].getVars(),
//                final_learning_rate, final_batch_size,
//                final_convergence_thresh, final_max_iterations,
//                1);


    }

    /**
     * Runs the RBF network at a variety of given learning rates, outputting
     * the results to console.
     */
    public static void tuneRBF_learning_rate() throws FileNotFoundException, UnsupportedEncodingException {
        String output = "../Output/" + "RBF_tune_learning_rate.csv";
        double[] learning_rates = {0.1, 0.01, 0.001, 0.0001};
        int k = 25;
        double batch_size = 0.10;
        double conv_thresh = 0.0005;
        int[] max_iter = {1000, 10000, 10000, 10000, 10000, 10000, 10000};
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
                        conv_thresh, max_iter[dataset],
                        folds);
            }
        }


    }

    /**
     * method to run the final configuration of a MLP network
     * @param datafiles
     * @param data
     */
    private static void finalMLP(String[] datafiles, DataReader[] data)
            throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println(" --- TESTING FINAL MLP CONFIG ---");

        String output_file = "../Output/" + "MLP-out.csv";
        // << Add funct to clear file contents here >>
        double final_learning_rate = 0.001;
        double final_momentum = 0.0;
        double final_batch_size = 0.1;
        double final_convergence_thresh = 0.0001;
        double final_hidden_nodes_mult = 1.0;         // multiply by the number of attributes to compute the number of hidden nodes
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
                        num_layers, final_hidden_nodes_mult, final_learning_rate,
                        final_batch_size, final_momentum, final_convergence_thresh,
                        final_max_iterations, final_num_folds);
            }
        }
    }

    /**
     * method to run the final configuration of a MLP network
     * @param datafiles
     * @param data
     */
    private static void tuneMLP(String[] datafiles, DataReader[] data)
            throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println(" --- TESTING FINAL MLP CONFIG ---");

        String output_file = "../Output/" + "MLP-out.csv";
        // << Add funct to clear file contents here >>
        double learning_rate = 0.1;
        double momentum = 0.0;
        double final_batch_size = 0.1;
        double final_convergence_thresh = 0.000001;
        double final_hidden_nodes_mult = 1.0;       // multiply by the number of attributes to compute the number of hidden nodes
        int final_max_iterations = 100000;
        int num_layers = 1;                         // for tuning, assume num_hidden layers is 1

        // iterate through data files
        for (int f = 0; f < data.length; f++) {
            // get current dataset
            DataReader curr_data = data[f];
            // run the network
            runMLP(output_file, datafiles[f], curr_data,
                    num_layers, final_hidden_nodes_mult, learning_rate,
                    final_batch_size, momentum, final_convergence_thresh,
                    final_max_iterations, 1);
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

            // Train and test the RBF network
            MLP mlp = new MLP(num_hidden_layers, num_hidden_nodes, learning_rate,
                                batch_size, momentum, convergence_threshold,
                                max_iterations);
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
        String output = data_set + "," + num_hidden_layers + "," + hidden_nodes_mult + "," + learning_rate + "," + metric1 + "," + metric2;
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
}
