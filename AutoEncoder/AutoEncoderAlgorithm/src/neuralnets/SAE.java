/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
import neuralnets.layer.Vector;
import java.io.FileNotFoundException;

/**
 *
 * @author natha
 */
public class SAE { // implements INeuralNet {
    
    private final int num_encoders;
    private final double sparsity_penalty;
    
    
    private final double learning_rate;
    private final double batch_size;
    private final double momentum;
    private final double convergence_threshold;
    private final int maximum_iterations;
    
    /** 
     * the final network that the SAE will build
     * this will consist of some autoencoding layers
     * as well as an MLP on top
    */
    private MLP network;
    
    /**
     * array to store the autoencoders while in training
     */
    private AutoEncoder[] encoders;
    
    public SAE(int num_encoders, double sparsity_penalty, double learning_rate, 
            double batch_size, double momentum, double convergence_threshold, 
            int max_iterations, SimilarityMatrix[] sim) {
        this.num_encoders = num_encoders;
        this.sparsity_penalty = sparsity_penalty;
        this.learning_rate = learning_rate;
        
        this.batch_size = batch_size;
        this.momentum = momentum;
        this.convergence_threshold = convergence_threshold;
        this.maximum_iterations = max_iterations;
        
        this.network = new MLP(learning_rate, batch_size, momentum, convergence_threshold, 
                                max_iterations, sim);
        
        this.encoders = new AutoEncoder[num_encoders];
    }

    /**
     * method to train a stacked auto encoder
     * This method will append layers to the end of the current network
     * 
     * If there are already trained auto encoders stored in a file,
     * call readExistingLayers(String fname) before training the network
     * @param training_set 
     */
    public void train(Set training_set) {
        // train next encoding layer
        System.out.println("-------------- TRAINING NEW ENCODING LAYER --------------");
        
        // train mlp with output of encoder
        //System.out.println("-------------- TRAINING PREDICTOR NETWORK --------------");
        
        // final training
        //System.out.println("-------------- PERFORMING FINAL NETWORK TRAINING --------------");
        
        // SHIFT THIS AROUND
        try { NetworkIO.writeLayers(network, "test.csv"); }
        catch (FileNotFoundException e) { System.err.println("error"); }
    }

    /**
     * method to test the predictions of the stacked auto encoder
     * @param testing_set
     * @return 
     */
    public double[] test(Set testing_set) {
        if (network.getNumLayers() == 0) { 
            System.err.println("No layers in network"); 
            return new double[] { -1.0 };
        }
        else { return this.network.test(testing_set); }
    }

    /**
     * method to compute the network's output for a given example
     * @param ex
     * @return 
     */
    public double predict(Example ex) {
        if (network.getNumLayers() == 0) { 
            System.err.println("No layers in network"); 
            return -1.0;
        }
        else { return this.network.predict(ex); }
    }
    
    /**
     * method to read in any existing encoding layers
     * @param network this should be a network with no layers
     * @param fname
     */
    public void readExistingLayers(String fname) {
        System.out.println("-------------- READING IN EXISTING AUTOENCODER --------------");
        // code for reading in network goes here
        System.out.println("-------------- READ IN " + this.network.getNumLayers() 
                + " LAYERS --------------");       
    }
    
    /**
     * method to encode the dataset using the auto encoder
     * 
     * this method should only be called before the final MLP
     * has been put on the top of the stacked Auto encoder
     * @param orig
     * @return 
     */
    private Set encode(Set orig) {
        if (this.network.getNumLayers() == 0) { return orig; }
        else {
            // compute number of attributes in encoded output
            Vector[] outputs = this.network.genLayerOutputs(orig.getExample(0));
            int last = outputs.length - 1;
            // create new set
            Set encoded = new Set(outputs[last].getLength(), orig.getNumClasses(), orig.getClassNames());
            // iterate through orig, encoding each example
            for (int i = 0; i < orig.getNumExamples(); i++) {
                Example ex = orig.getExample(i);
                Vector temp = this.network.genLayerOutputs(ex)[last];
                encoded.addExample(new Example(ex.getValue(), ex.getSubsetIndex(), temp));
            }
            return encoded;
        }
    }
    
}
