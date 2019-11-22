/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package client;

import poptrain.PSO;

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
    public static void main(String[] args) {
        
        // READ IN DATA
        for(int i = 0; i < data.length; i++) { data[i] = new DataReader(datafiles[i]); }
        
        int input_dim = data[0].getSubsets()[0].getNumAttributes();
        int output_dim = data[0].getSubsets()[0].getNumClasses();
        PSO pso = new PSO(new int[] { 1, 2, input_dim, output_dim }, data[0].getSimMatrices());
    }
    
}
