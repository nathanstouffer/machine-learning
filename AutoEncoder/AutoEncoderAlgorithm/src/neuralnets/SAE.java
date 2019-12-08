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
import java.io.IOException;
import neuralnets.layer.Layer;

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
    private SimilarityMatrix[] sim;
    
    /** 
     * the final network that the SAE will build
     * this will consist of some auto encoding layers
     * as well as an MLP on top
    */
    private MLP network;
    
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
        this.sim = sim;
        
        this.network = new MLP(learning_rate, batch_size, momentum, convergence_threshold, 
                                max_iterations, sim);
    }

    /**
     * method to train a stacked auto encoder
     * This method will append layers to the end of the current network
     * 
     * If there are already trained auto encoders stored in a file,
     * call readExistingLayers(String fname) before training the network
     * @param training_set 
     */
    public void train(Set training_set) throws FileNotFoundException {
        SimilarityMatrix[] temp_sim = new SimilarityMatrix[] {};
        if (this.network.getNumLayers() == 0) { temp_sim = this.sim; }
        // train next encoding layer
        System.out.println("-------------- TRAINING NEW ENCODING LAYER --------------");
        // encode examples with current autoencoder state
        Set encoded = this.encode(training_set);
        
        // create a new autoencoder
        AutoEncoder ae = new AutoEncoder(this.sparsity_penalty, this.learning_rate,
                            this.batch_size, this.momentum, this.convergence_threshold,
                            this.maximum_iterations, temp_sim);
        ae.train(encoded);
        
        // add new encoding layer to network
        Layer layer = ae.getLayer(0);
        this.network.addLayer(layer);
        
        // write new layers to a file
        String fname = training_set.getDataSetName() + "-" + this.network.getNumLayers() 
                        + "-layer-ae.csv";
        NetworkIO.writeLayers(this.network, fname);
        
        System.out.println("-------------- TRAINING PREDICTOR NETWORK --------------");
        // re-encode training data with new layer
        encoded = this.encode(training_set);
        MLP temp = new MLP(1, new int[] {2*encoded.getNumAttributes()}, this.learning_rate,
                            this.batch_size, this.momentum, this.convergence_threshold,
                            this.maximum_iterations, new SimilarityMatrix[] {});
        temp.train(encoded);
        
        // add in prediction layer
        for (int l = 0; l < temp.getNumLayers(); l++) { 
            this.network.addLayer(temp.getLayer(l)); 
        }
        
        // final training
        System.out.println("-------------- PERFORMING FINAL NETWORK TRAINING --------------");
        this.network.train(training_set);
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
    public void readExistingLayers(String fname) throws IOException {
        System.out.println("-------------- READING IN EXISTING AUTOENCODER --------------");
        
        Layer[] temp = NetworkIO.readLayers(fname);
        for (int t = 0; t < temp.length; t++) {
            this.network.addLayer(temp[t]);
        }
        
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
