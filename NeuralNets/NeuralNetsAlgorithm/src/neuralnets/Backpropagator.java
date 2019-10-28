/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import networklayer.*;
import datastorage.*;

/**
 * class that implements the backpropagation algorithm for
 * a neural network
 * 
 * @author natha
 */
public class Backpropagator {
    
    // current state of the weights in the network
    private final INeuralNet network;
    // gradient for the weights in the network
    private Matrix[] gradient;
    
    /**
     * constructor to initialize global variables
     * @param network 
     */
    public Backpropagator(INeuralNet network) {
        // instantiate global variables
        this.network = network;
        
        int[][] layer_dim = this.network.getLayerDim();
        // instantiate gradient to correct size
        this.gradient = new Matrix[layer_dim.length];
        // initialize gradient values to 0.0
        for (int i = 0; i < this.gradient.length; i++) {
            int num_rows = layer_dim[i][0];         // num_rows is in 0th pos
            int num_col = layer_dim[i][1];          // num_col is in 1st pos
            this.gradient[i] = new Matrix(num_rows, num_col);
        }
    }
    
    /**
     * method to compute the average gradient of a network
     * with respect to the argument batch
     * 
     * @param batch must be set
     * @return 
     */
    public Matrix[] computeGradient(Set batch) {
        // update gradient for each example
        for (int i = 0; i < batch.getNumExamples(); i++) {
            Example ex = batch.getExample(i);
            // generate output using current network
            double actual = ex.getValue();
            
            // store the layer outputs for the current example
            // outputs should be indexed at +1 than normal
            Vector[] outputs = this.network.genLayerOutputs(ex);
            // store the derivative of outputs for current example
            Vector[] derivatives = this.network.genLayerDeriv();
            
            // index variable for the current layer
            int layer = derivatives.length;
            
            // identify correct output
            Vector target = new Vector(outputs[layer+1].getLength());
            if (target.getLength() == 1) { target.set(0, actual); }         // if the len(output) is 1, then this is a regression data set
            else {
                // we now deal with classification
                int class_index = (int)actual;
                // iterate through the target vector
                for (int j = 0; j < target.getLength(); j++) {
                    if (j == class_index) { target.set(j, 1.0); }           // set target to 1.0 if at correct class
                    else { target.set(j, 0.0); }                            // otherwise set to 0.0
                }
            }
            
            // get output of current layer
            Vector output = outputs[layer+1];
            // get derivatives of output layer
            Vector deriv = derivatives[layer];
            // compute current deltas
            Vector deltas = new Vector(target.getLength());
            for (int j = 0; j < deltas.getLength(); j++) {
                // compute delta
                double delta = output.get(j) - target.get(j);
                delta = deriv.get(j) * delta;
                
                // add value to deltas vector
                deltas.set(j, delta);
            }

            // update gradient
            this.updateGradient(this.gradient[layer], deltas, outputs[layer]);
            
            // propagate backwards through remaining layers
            this.backpropagate(layer-1, outputs, derivatives, deltas);
        }
        
        // average gradient over the size of the batch
        for (int i = 0; i < this.gradient.length; i++) { this.gradient[i].divEquals(batch.getNumExamples()); }
        
        // return gradient
        return this.gradient;
    }
    
    /**
     * method to recursively propagate back through the network, updating
     * the gradient for each weight
     * 
     * @param layer
     * @param ds_deltas 
     */
    private void backpropagate(int layer, Vector[] outputs, Vector[] derivatives, Vector ds_deltas) {
        // can not propagate past inputs
        if (layer >= 0) {
            Matrix curr_layer = this.gradient[layer];
            Vector deriv = derivatives[layer];
            // compute deltas
            Vector deltas = new Vector(curr_layer.getNumCol());
            for (int i = 0; i < deltas.getLength(); i++) {
                // initialize delta to 0.0
                double delta = 0.0;
                // sum the effects of delta in next layer
                for (int j = 0; j < ds_deltas.getLength(); j++) {     // DOUBLE CHECK THIS LOOP
                    Vector row = curr_layer.getRow(j);
                    delta += ds_deltas.get(j) * row.get(i);
                }
                delta *= deriv.get(i);
            }
            
            // update gradient
            this.updateGradient(this.gradient[layer], deltas, outputs[layer]);
            
            // recursive call
            this.backpropagate(layer-1, outputs, derivatives, ds_deltas);
        }
    }
    
    /**
     * method to update the gradient for the current layer
     * @param deltas
     * @param input 
     */
    private void updateGradient(Matrix layer, Vector deltas, Vector input) {
        // iterate through rows
        for (int i = 0; i < layer.getNumRows(); i++) {
            Vector row = layer.getRow(i);
            // compute row update
            Vector update = new Vector(input.getLength());
            for (int j = 0; j < input.getLength(); j++) { 
                double val = deltas.get(i) * input.get(j);
                update.set(j, val);
            }
            // add to row
            row.plusEquals(update);
        }
    }
    
}
