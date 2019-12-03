/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package client;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import neuralnets.MLP;
import poptrain.PSO;
import java.util.Arrays;
import poptrain.GeneticAlgorithm;

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
    public static void main(String[] args) throws FileNotFoundException {
        // READ IN DATA
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }
        
        // ------------------------------------------------------------
        // --- RUN FINAL GA TESTS WITH OPTIMAL PARAMETERS SELECTED ----
        // ------------------------------------------------------------
        tuneGA();
        //finalGA();
        
        // ------------------------------------------------------------
        // --- RUN FINAL PSO TESTS WITH OPTIMUM PARAMETERS SELECTED ---
        // ------------------------------------------------------------
        //tunePSO();
        finalPSO();
    }
    /**
     * private method to tune GA
     */
    private static void tuneGA() throws FileNotFoundException {
        System.out.println("--------- TUNING GA CONFIG ---------");

        String fout = "../Output/" + "GA-tuning-out.csv";
        clearFile(fout);
        
        // final configuration of variables listed here
        double[] crossover = { 0.1, 0.05, 0.01, 0.005};
        double[] mutation = { 0.05, 0.02, 0.01, 0.005};
        int pop_size = 64;
        int max_iter = 1000;
        int folds = 1;
        
        int num_hl = 1;
        // iterate through data files
        for (int f = 3; f < 6; f++) {//data.length; f++) {
            // iterate through crossover rates
            for (int c = 0; c < crossover.length; c++) {
                // iterate through mutation rates
                for (int m = 0; m < mutation.length; m++) {
                    runGA(fout, data[f], num_hl, crossover[c],
                            mutation[m], pop_size, max_iter, folds);
                }
            }
        }
    }
    
    
    /**
     * private method to run GA with the final configuration 
     * which is specified in this method
     */
    private static void finalGA() throws FileNotFoundException {
        System.out.println("--------- TESTING FINAL GA CONFIG ---------");

        String fout = "../Output/" + "GA-final-out.csv";
        clearFile(fout);
        
        // final configuration of variables listed here
        double crossover_rate = 0.01;
        double mutation_rate = 0.01;
        int pop_size = 104;
        int max_iter = 100;
        int folds = 1;
        
        // FOR TESTING ONLY
        int TODO = 0;
        int num_hl = 0;
        runGA(fout, data[TODO], num_hl, crossover_rate, mutation_rate, pop_size, max_iter, folds);
        
        /*// iterate through data files
        for (int f = 0; f < data.length; f++) {
            // iterate through number of layers
            for (int num_hl = 0; num_hl < 3; num_hl++) {
                runPSO(fout, data[f], num_hl, cog_mult,
                        soc_mult, pop_size, max_iter, folds);
            }
        }
        */
    }
    
    /**
     * 
     * @param fout
     * @param data
     * @param num_hl
     * @param crossover_rate
     * @param mutation_rate
     * @param pop_size
     * @param max_iter
     * @param folds
     * @throws FileNotFoundException 
     */
    private static void runGA(String fout, DataReader data, int num_hl,
            double crossover_rate, double mutation_rate, int pop_size, int max_iter, 
            int folds) throws FileNotFoundException {
        System.out.println("---- RUNNING GA ON DATASET " + data.getFileName() + " WITH " + num_hl 
                + " HIDDEN LAYERS ----");
        System.out.print("------ POP SIZE: " + pop_size);
        System.out.print(" ---- Pc: " + crossover_rate);
        System.out.println(" ---- Pm: " + mutation_rate + " ------");
        
        double starttime = System.currentTimeMillis();
        
        double metric1 = 0;
        double metric2 = 0;
        
        // build topology array
        int[] topology = buildTop(data, num_hl);
        
        // instantiate GA class
        GeneticAlgorithm ga = new GeneticAlgorithm(topology, max_iter, pop_size, crossover_rate, mutation_rate, data.getSimMatrices());
        
        // perform cross validation
        for (int c = 0; c < folds; c++) {
            System.out.println("Performing CV Fold #" + (c+1));
            
            Set training = new Set(data.getSubsets(), c);
            Set testing = data.getSubsets()[c];
            
            // train the swarm
            ga.train(training);
            // get most fit member
            MLP mlp = ga.getBest();
            // test mlp
            double[] results = mlp.test(testing);
            
            // Get metrics
            if(training.getNumClasses() == -1) {
                // Regression
                RegressionEvaluator eval = new RegressionEvaluator(results, testing);
                metric1 += eval.getMSE();
                metric2 += eval.getME();
                eval.printAct();
                eval.printPred();
            } else {
                // Classification
                ClassificationEvaluator eval = new ClassificationEvaluator(results, testing);
                metric1 += eval.getAccuracy();
                metric2 += eval.getMSE();
                eval.printAct();
                eval.printPred();
            }
        }
        
        // average metrics
        metric1 /= folds;
        metric2 /= folds;
        
        // output results
        String output = data.getFileName() + "," + pop_size + "," + num_hl + "," + crossover_rate + "," 
                + mutation_rate + "," + metric1 + "," + metric2;
        // write to console
        double endtime = System.currentTimeMillis();
        double runtime = (endtime - starttime) / 1000;
        System.out.println("\u001B[33m" + "GA trained and tested in " + runtime + " seconds");
        System.out.println(output + "\u001B[0m");
        // Write to file
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fout), true /* append = true */));
        writer.println(output);
        writer.close();
    }
    
    /**
     * private method to run PSO with the final configuration 
     * which is specified in this method
     */
    private static void tunePSO() throws FileNotFoundException {
        System.out.println("--------- TUNING PSO CONFIG ---------");

        String fout = "../Output/" + "PSO-tuning-out.csv";
        clearFile(fout);
        
        // final configuration of variables listed here
        double[] cog_mult = { 1.0, 2.0, 3.0 };
        double[] soc_mult = { 1.0, 2.0, 3.0 };
        int pop_size = 100;
        int max_iter = 1000;
        int folds = 1;
        
        int num_hl = 1;
        // iterate through data files
        for (int f = 4; f < data.length; f++) {//data.length; f++) {
            // iterate through cog mult values
            for (int c = 0; c < cog_mult.length; c++) {
                // iterate through soc mult values
                for (int s = 0; s < soc_mult.length; s++) {
                    runPSO(fout, data[f], num_hl, cog_mult[c],
                            soc_mult[s], pop_size, max_iter, folds);
                }
            }
        }
    }
    
    /**
     * private method to run PSO with the final configuration 
     * which is specified in this method
     */
    private static void finalPSO() throws FileNotFoundException {
        System.out.println("--------- TESTING FINAL PSO CONFIG ---------");

        String fout = "../Output/" + "PSO-final-out.csv";
        clearFile(fout);
        
        // final configuration of variables listed here
        double cog_mult = 3;
        double soc_mult = 1;
        int pop_size = 100;
        int max_iter = 1000;    // MAYBE TRY FINAL RUN WITH MORE INDIVIDUALS IN THE POPULATION?
        int folds = 1;
        
        // FOR TESTING ONLY
        int TODO = 4;
        int num_hl = 1;
        runPSO(fout, data[TODO], num_hl, cog_mult, soc_mult, pop_size, max_iter, folds);
        
        /*// iterate through data files
        for (int f = 0; f < data.length; f++) {
            // iterate through number of layers
            for (int num_hl = 0; num_hl < 3; num_hl++) {
                runPSO(fout, data[f], num_hl, cog_mult,
                        soc_mult, pop_size, max_iter, folds);
            }
        }
        */
    }
    
    /**
     * private method to build a run of PSO with the specified parameters
     * @param fout
     * @param data_set
     * @param data
     * @param num_hl
     * @param cog_mult
     * @param soc_mult
     * @param pop_size
     * @param max_iter
     * @param folds 
     */
    private static void runPSO(String fout, DataReader data, int num_hl,
            double cog_mult, double soc_mult, int pop_size, int max_iter, 
            int folds) throws FileNotFoundException {
        System.out.println("---- RUNNING PSO ON DATASET " + data.getFileName() + " WITH " + num_hl 
                + " HIDDEN LAYERS ----");
        System.out.print("------ POP SIZE: " + pop_size);
        System.out.print(" ---- COG MULT: " + cog_mult);
        System.out.println(" ---- SOC MULT: " + soc_mult + " ------");
        
        double starttime = System.currentTimeMillis();
        
        double metric1 = 0;
        double metric2 = 0;
        
        // build topology array
        int[] topology = buildTop(data, num_hl);
        
        // instantiate PSO class
        PSO pso = new PSO(cog_mult, soc_mult, topology, pop_size, max_iter, data.getSimMatrices());
        
        // perform cross validation
        for (int c = 0; c < folds; c++) {
            System.out.println("Performing CV Fold #" + (c+1));
            
            Set training = new Set(data.getSubsets(), c);
            Set testing = data.getSubsets()[c];
            
            // train the swarm
            pso.train(training);
            // get most fit member
            MLP mlp = pso.getBest();
            // test mlp
            double[] results = mlp.test(testing);
            
            // Get metrics
            if(training.getNumClasses() == -1) {
                // Regression
                RegressionEvaluator eval = new RegressionEvaluator(results, testing);
                metric1 += eval.getMSE();
                metric2 += eval.getME();
                eval.printAct();
                eval.printPred();
            } else {
                // Classification
                ClassificationEvaluator eval = new ClassificationEvaluator(results, testing);
                metric1 += eval.getAccuracy();
                metric2 += eval.getMSE();
                eval.printAct();
                eval.printPred();
            }
        }
        
        // average metrics
        metric1 /= folds;
        metric2 /= folds;
        
        // output results
        String output = data.getFileName() + "," + pop_size + "," + num_hl + "," + cog_mult + "," 
                + soc_mult + "," + metric1 + "," + metric2;
        // write to console
        double endtime = System.currentTimeMillis();
        double runtime = (endtime - starttime) / 1000;
        System.out.println("\u001B[33m" + "PSO trained and tested in " + runtime + " seconds");
        System.out.println(output + "\u001B[0m");
        // Write to file
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(fout), true /* append = true */));
        writer.println(output);
        writer.close();
    }
    
    /**
     * private static method to build the topology of a neural network
     * @param data
     * @param num_hl
     * @return 
     */
    private static int[] buildTop(DataReader data, int num_hl) {
        // number of hidden layers
        int len = num_hl + 3;
        int[] topology = new int[len];
        topology[0] = num_hl;
        // populate input and output dimensions
        Set temp = data.getSubsets()[0];
        topology[len-2] = computeInputDim(temp.getExample(0), data.getSimMatrices());
        if (temp.getNumClasses() == -1) { topology[len-1] = 1; }
        else { topology[len-1] = temp.getNumClasses(); }
        
        // populate hidden layer multiples
        for (int t = 1; t < num_hl+1; t++) { topology[t] = 2 * temp.getNumAttributes(); }       
        return topology;
    }
    
    /**
     * method to compute the dimensions of the input layer
     * @return 
     */
    private static int computeInputDim(Example temp, SimilarityMatrix[] sim) {
        int input_dim = 0;
        if (sim.length == 0) { input_dim = temp.getAttributes().size(); }
        else {
            int s = 0;          // index for similarity matrix
            // iterate through attributes
            for (int i = 0; i < temp.getAttributes().size(); i++) {
                // check for indexing error
                if (s < sim.length) {
                    // test for categorical attribute and add appropriate number
                    if (i == sim[s].getAttrIndex()) {
                        input_dim += sim[s].getNumOptions();
                        s++;
                    }
                    else { input_dim++; }
                }
                // otherwise add 1
                else { input_dim++; }
            }
        }
        return input_dim;
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
