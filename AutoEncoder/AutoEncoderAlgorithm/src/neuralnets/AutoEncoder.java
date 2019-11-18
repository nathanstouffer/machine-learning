/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
import neuralnets.layer.Layer;
import neuralnets.layer.Linear;
import neuralnets.layer.Logistic;
import neuralnets.layer.Vector;
import java.util.ArrayList;
import neuralnets.layer.Matrix;

/**
 * class that implements an AutoEncoder. The Auto encoder is trained using
 * Back propagation
 * 
 * 
 * @author natha
 */
public class AutoEncoder implements INeuralNet {
    
    private final double STARTING_WEIGHT_BOUND = 0.0001;
    private final double IMPACT_THRESH = 0.0075;
    
    /**
     * The sparsity penalty is the defining feature of an over complete auto encoder
     * The is the penalty we apply to the weights that go to the output layer so that
     * they tend toward 0
     */
    private final double sparsity_penalty;
    
    /**
     * The learning rate for the network is a tunable parameter that affects
     * the impact of each iteration of weight updates on the output layer.
     * Batch size is the percentage of examples in each training set to use
     * at one time while applying gradient descent to the output layer.
     */
    private final double learning_rate;
    private final double batch_size;
    
    /**
     * The convergence threshold and maximum iterations determine the 
     * termination characteristics during training. Training the layer
     * weights will terminate when the gradient updates are all weighted less
     * than the convergence threshold (as a percentage) multiplied by the
     * current weights or when the number of training iterations reaches the
     * specified maximum.
     */
    private final double convergence_threshold;
    private final int maximum_iterations;
    
    /**
     * momentum helps the network train faster. It moves the gradient step
     * in the loss function in the direction that the previous iterations 
     * gradient descent path. Momentum is a number to scale the magnitude
     * of that step
     */
    private final double momentum;
    
    private final SimilarityMatrix[] sim;
    
    /**
     * The array of layers holds the weights that make up the network.
     * Feeding an example into the layers produces a prediction
     */
    private Layer[] layers;
    private final int HIDDEN_NODES_MULT = 2;
    
    public AutoEncoder(double sparsity_penalty, double _learning_rate, 
            double _momentum, double _batch_size, double _convergence_threshold, 
            int _maximum_iterations, SimilarityMatrix[] _sim) {
        this.sparsity_penalty = sparsity_penalty;
        this.learning_rate = _learning_rate;
        this.batch_size = _batch_size;
        this.momentum = _momentum;
        this.layers = new Layer[2];
        this.convergence_threshold = _convergence_threshold;
        this.maximum_iterations = _maximum_iterations;
        this.sim = _sim;
    }

    @Override
    public void train(Set training_set) {
        // compute input dimensions
        int input_dim = this.computeInputDim(training_set.getExample(0));
        layers[0] = new Layer(new Logistic(), HIDDEN_NODES_MULT*input_dim, input_dim+1);
        layers[1] = new Layer(new Linear(), input_dim, HIDDEN_NODES_MULT*input_dim+1);
        //layers[1] = new Layer(new Logistic(), input_dim, HIDDEN_NODES_MULT*input_dim+1);
        // UNSURE IF THIS IS LOGISTIC OR LINEAR
        
        // randomly initialize weights
        for (int i = 0; i < layers.length; i++) {
            layers[i].randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        }
        
        // construct a backpropagator to train the network
        Backpropagator backprop = new Backpropagator(this);
        
        boolean converged = false;
        int iterations = 0;
        // initialize prev_gradient to null
        Matrix[] prev_gradient = null;
        while (iterations < this.maximum_iterations && !converged) {
            // compute a batch
            Set batch = training_set.getRandomBatch(this.batch_size);
            // compute the gradient
            Matrix[] gradient = backprop.computeGradient(batch);
            // multiply gradient by the learning rate
            for (int i = 0; i < gradient.length; i++) { gradient[i].timesEquals(this.learning_rate); }
            // apply gradient
            for (int i = 0; i < layers.length; i++) { layers[i].plusEquals(gradient[i]); }
            
            // apply momentum if necessary
            if (momentum != 0.0 && prev_gradient != null) {
                // multiply previous gradient by momentum rate
                for (int i = 0; i < prev_gradient.length; i++) { prev_gradient[i].timesEquals(this.learning_rate); }
                // apply momentum
                for (int i = 0; i < layers.length; i++) { this.layers[i].plusEquals(prev_gradient[i]); }
            }
            // update the previous gradient
            prev_gradient = gradient;
            
            // Output progress to console and check for convergence
            if(iterations % (maximum_iterations/100) == 0) {
                System.out.println("-> Training Autoencoder network iteration: " + iterations);
                converged = hasConverged(gradient, true); // Verbose to print status
            } else {
                converged = hasConverged(gradient, false);
            }
            iterations++;
        }
        
        this.prune();
    }

    /**
     * public method to test the current state of the auto encoder.
     * This is meant to be used before cutting off the final layer
     * @param testing_set
     * @return 
     */
    public Vector[] testAutoEncoder(Set testing_set) {
        Vector[] outputs = new Vector[testing_set.getNumExamples()];
        //Set outputs = new Set(testing_set.getNumAttributes(), testing_set.getNumClasses(), testing_set.getClassNames());
        ArrayList<Example> examples = testing_set.getExamples();
        for (int i = 0; i < testing_set.getNumExamples(); i++) {
            //Example recon = this.computeReconstructedOutput(examples.get(n));
            //outputs.addExample(recon);
            outputs[i] = this.computeReconstructedOutput(testing_set.getExample(i));
        }
        return outputs;
    }
    
    /**
     * method to compute the output of the full auto encoder
     * This is used when training the auto encoder
     * @param ex
     * @return 
     */
    protected Vector computeReconstructedOutput(Example ex) { 
        return this.genLayerOutputs(ex)[2];
        //Vector output = this.genLayerOutputs(ex)[2];
        //return new Example(ex.getValue(), ex.getSubsetIndex(), output, this.sim);
    }
    
    /**
     * Returns the output of each layer
     *
     * @param ex input example
     * @return A vector array containing the output of each layer.
     */
    @Override
    public Vector[] genLayerOutputs(Example ex) {
        Vector[] outputs = new Vector[layers.length + 1];
        outputs[0] = new Vector(ex, sim);
        for (int i = 0; i < layers.length; i++) {
            outputs[i].insertBiasMultiplier();
            outputs[i + 1] = layers[i].feedForward(outputs[i]);
        }
        return outputs;
    }

    /**
     * Returns the derivative of each layer Note: genLayerOutputs() must be
     * called prior to this function call, or there will be no current
     * derivative.
     *
     * @return vector array where index is layer
     */
    @Override
    public Vector[] genLayerDeriv() {
        Vector[] deriv = new Vector[layers.length];
        for (int i = 0; i < layers.length; i++) {
            deriv[i] = layers[i].getDeriv();
        }
        return deriv;
    }

    /**
     * Returns the dimensions of the layers.
     *
     * @return A two dimensional int array. First index is layer. Second index
     * is: 0 -> # nodes and 1 -> # inputs to the layer.
     */
    @Override
    public int[][] getLayerDim() {
        // calculate dimesions for all layers
        int[][] dim = new int[layers.length][2];
        for (int i = 0; i < layers.length; i++) {
            dim[i][0] = layers[i].getNumNodes();
            dim[i][1] = layers[i].getNumInputs();
        }
        return dim;
    }
    
    /**
     * method to compute the dimensions of the input layer
     * @return 
     */
    private int computeInputDim(Example temp) {
        int input_dim = 0;
        if (this.sim.length == 0) { input_dim = temp.getAttributes().size(); }
        else {
            int s = 0;          // index for similarity matrix
            // iterate through attributes
            for (int i = 0; i < temp.getAttributes().size(); i++) {
                // check for indexing error
                if (s < this.sim.length) {
                    // test for categorical attribute and add appropriate number
                    if (i == this.sim[s].getAttrIndex()) {
                        input_dim += this.sim[s].getNumOptions();
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
     * Check if the network has converged. Will return true if no values within
     * the gradient are higher than CONVERGENCE_THRESHOLD (a percentage) of the
     * existing layer weights.
     * @return
     */
    private boolean hasConverged(Matrix[] gradient, boolean verbose) {
        // iterate through layers
        for (int i = 0; i < gradient.length; i++) { 
            // get weights and gradient for current layer
            Matrix weights = this.layers[i].getWeights();
            Matrix curr_gradient = gradient[i];
            // Iterate through rows
            for(int row = 0; row < weights.getNumRows(); row++) {
                Vector grad_row = curr_gradient.getRow(row);
                Vector weight_row = weights.getRow(row);
                // Iterate through cols
                for(int col = 0; col < weights.getNumCol(); col++) {
                    double g = Math.abs(grad_row.get(col));
                    double w = Math.abs(weight_row.get(col));
                    // Test if the value exceeds the threshol
                    if( g > w*convergence_threshold) {
                        if(verbose) {
                            //System.out.println("    ---> GRADIENT TO WEIGHT RATIO = " + (g/w) + " WHERE THRESHOLD = " + convergence_threshold);
                            //System.out.println("    ---> W = " + w + " G = " + g);
                        }
                        return false;
                    }
                }
            }
        }
        // if we made it through the loops, the weights have converged
        return true;
    }
    
    /**
     * private method to prune out unnecessary hidden nodes
     * 
     * this is done by computing the average (absolute) value of 
     * weights leaving that node, and then comparing that to an impact
     * threshold
     */
    private void prune() {
        // get weights to the output layer
        Matrix last = this.layers[1].getWeights();
        // indicative array of whether a node should be kept
        int[] to_rm = new int[last.getNumCol()];
        // store number of nodes we will keep
        int kept = 0;
        // identify which nodes to prune
        for (int c = 1; c < last.getNumCol(); c++) {
            double avg = 0.0;
            // iterate through rows, computing average
            for (int r = 0; r < last.getNumRows(); r++) {
                avg += Math.abs(last.getRow(r).get(c));
            }
            avg /= last.getNumRows();
            
            if (avg > this.IMPACT_THRESH) { to_rm[c] = 0; kept++; }
            else { to_rm[c] = 1; }
        }
        
        System.out.println("\nKEPT " + kept + " OF " + to_rm.length + " HIDDEN NODES\n");
        
        // iterate through to_rm array
        int num_del = 0;
        // start at 1 to ignore bias node
        for (int n = 1; n < to_rm.length; n++) {
            if (to_rm[n] == 1) {
                // we must remove the nth hidden node
                this.layers[0].delRow(n - num_del - 1); // FOR NO BIAS NODE
                this.layers[1].delCol(n - num_del);
                num_del++;
            }
        }
    }
    
    /**
     * method to return the layer at the specified index
     * @param index
     * @return 
     */
    public Layer getLayer(int index) { return this.layers[index]; }
    
    public SimilarityMatrix[] getSimMtx() { return this.sim; }
    public int getNumLayers() { return this.layers.length; }
    public double getSparsityPenalty() { return this.sparsity_penalty; }
    
    @Override
    public double[] test(Set testing_set) {
        throw new UnsupportedOperationException("Not supported for Autoencoder"); 
    }

    @Override
    public double predict(Example ex) { 
        throw new UnsupportedOperationException("Not supported for Autoencoder"); 
    }
}
