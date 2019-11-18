/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import neuralnets.layer.Vector;
import neuralnets.layer.Matrix;
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
        // clear current gradient
        for (int i = 0; i < this.gradient.length; i++) { this.gradient[i].clear(); }
        
        // update gradient for each example
        for (int i = 0; i < batch.getNumExamples(); i++) {
            Example ex = batch.getExample(i);
            
            // store the layer outputs for the current example
            // outputs should be indexed at +1 than normal
            Vector[] outputs = this.network.genLayerOutputs(ex);
            // store the derivative of outputs for current example
            Vector[] derivatives = this.network.genLayerDeriv();
            
            // index variable for the current layer
            int layer = derivatives.length - 1;
            // compute target
            Vector target = this.computeTarget(ex, outputs[layer+1].getLength());
            
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
            
            // if we are dealing with an autoencoder, make the weights
            // to the output layer sparse
            this.sparsify();
        }
        
        // average gradient over the size of the batch
        for (int i = 0; i < this.gradient.length; i++) { this.gradient[i].divEquals(batch.getNumExamples()); }
        
        // multiply gradient by -1 for minimizing loss function
        for (int i = 0; i < this.gradient.length; i++) { this.gradient[i].timesEquals(-1); }
        
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
            Matrix curr_layer = this.network.getLayer(layer).getWeights();
            Matrix next_layer = this.network.getLayer(layer+1).getWeights();
            Vector deriv = derivatives[layer];
            // compute deltas
            Vector deltas = new Vector(curr_layer.getNumRows());
            for (int j = 0; j < deltas.getLength(); j++) {
                // initialize delta to 0.0
                double delta = 0.0;
                // sum the effects of delta from next layer
                for (int k = 0; k < ds_deltas.getLength(); k++) { 
                    Vector row = next_layer.getRow(k);
                    delta += ds_deltas.get(k) * row.get(j);
                }
                delta *= deriv.get(j);
                deltas.set(j, delta);
            }
            
            // update gradient
            this.updateGradient(this.gradient[layer], deltas, outputs[layer]);
            
            // recursive call
            this.backpropagate(layer-1, outputs, derivatives, deltas);
        }
    }
    
    /**
     * method to update the gradient for the current weights
     * @param deltas
     * @param input 
     */
    private void updateGradient(Matrix weights, Vector deltas, Vector input) {
        // iterate through nodes
        for (int i = 0; i < weights.getNumRows(); i++) {
            Vector row = weights.getRow(i);
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
    
    /**
     * method to compute the target of 
     * @param ex
     * @return 
     */
    private Vector computeTarget(Example ex, int size) {
        // identify type of network
        if (this.network.getSparsityPenalty() != 0.0) {
            // network is an autoencoder
            return new Vector(ex, this.network.getSimMtx());
        }
        else {
            // we now know that we are dealing with classification or regression
            double actual = ex.getValue();
            // identify correct output
            Vector target = new Vector(size);
            if (target.getLength() == 1) { target.set(0, actual); }     // if the len(output) is 1, then this is a regression data set
            else {
                // we now deal with classification
                int class_index = (int)actual;
                // iterate through the target vector
                for (int j = 0; j < target.getLength(); j++) {
                    if (j == class_index) { target.set(j, 1.0); }       // set target to 1.0 if at correct class
                    else { target.set(j, 0.0); }                        // otherwise set to 0.0
                }
            }
            return target;
        }
    }
    
    /**
     * private method to apply penalty to the weights
     * that go to the output layer of an auto encoder
     */
    private void sparsify() {
        // last index of gradient, this corresponds to the network
        int last = this.gradient.length - 1;
        // value of sparsity penalty
        double sparsity_penalty = this.network.getSparsityPenalty();
        // matrix of the current state of the weights to the output layer
        Matrix output_weights = this.network.getLayer(last).getWeights();
        // empty penalty matrix
        Matrix penalty = new Matrix(output_weights.getNumRows(), output_weights.getNumCol());
        // populate the penalty matrix
        for (int i = 0; i < output_weights.getNumRows(); i++) {
            // the populate is done row by row 
            Vector weight_row = output_weights.getRow(i);
            Vector penalty_row = new Vector(weight_row.getLength());
            for (int j = 0; j < output_weights.getNumCol(); j++) {
                double weight = weight_row.get(j);
                if (weight < 0.0) { penalty_row.set(j, -1.0 * sparsity_penalty); }
                else { penalty_row.set(j, sparsity_penalty); }
            }
            penalty.setRow(i, penalty_row);
        }
        // add penalty to last layer in gradient
        this.gradient[last].plusEquals(penalty);
    }
    
}
