/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
import java.util.ArrayList;
import networklayer.Layer;
import networklayer.Linear;
import networklayer.Logistic;
import networklayer.Matrix;
import networklayer.Vector;

/**
 *
 * @author Kevin
 */
public class MLP implements INeuralNet {
    
    /**
     * The absolute value of the bounds on the starting weights in the output
     * layer.
     */
    private static final double STARTING_WEIGHT_BOUND = 0.0001;

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
     * in the loss function in the direction that the previous iteration's 
     * gradient descent path. Momentum is a number to scale the magnitude
     * of that step
     */
    private final double momentum;
    
    /**
     * storing the similarity matrix for categorical variables
     */
    private final SimilarityMatrix[] sim;

    /**
     * The array of layers holds the weights that make up the network.
     * Feeding an example into the layers produces a prediction
     */
    private Layer[] layers;
    private final int num_hidden_layer;
    private final int[] num_hidden_nodes;

    public MLP(int _num_hidden_layers, int[] _num_hidden_nodes, double _learning_rate, 
            double _batch_size, double _momentum, double _convergence_threshold, 
            int _maximum_iterations, SimilarityMatrix[] _sim) {
        num_hidden_layer = _num_hidden_layers;
        num_hidden_nodes = new int[num_hidden_layer];
        for (int i = 0; i < num_hidden_nodes.length; i++) { num_hidden_nodes[i] = _num_hidden_nodes[i]; }
        learning_rate = _learning_rate;
        batch_size = _batch_size;
        momentum = _momentum;
        layers = new Layer[num_hidden_layer + 1];
        convergence_threshold = _convergence_threshold;
        maximum_iterations = _maximum_iterations;
        sim = _sim;
    }

    @Override
    public void train(Set training_set) { 
        // compute input dimensions
        int input_dim = this.computeInputDim(training_set.getExample(0));
        this.initializeLayers(training_set.getNumClasses(), input_dim);

        // Create a backpropagator that will just train the output layer.
        Backpropagator backprop = new Backpropagator(this);

        boolean converged = false;
        int iterations = 0;
        // initialize prev_gradient to null
        Matrix[] prev_gradient = null;
        while (iterations < maximum_iterations && !converged) {
            // Compute a batch
            Set batch = training_set.getRandomBatch(batch_size);
            // Get gradient
            Matrix[] gradient = backprop.computeGradient(batch);
            // Multiply gradient by learning rate
            for (int k = 0; k < gradient.length; k++) { gradient[k].timesEquals(learning_rate); }
            // apply gradient
            for (int k = 0; k < layers.length; k++) { layers[k].plusEquals(gradient[k]); }
                
            // apply momentum if necessary
            if (momentum != 0.0 && prev_gradient != null) {
                // multiply by momentum rate
                for (int k = 0; k < prev_gradient.length; k++) { prev_gradient[k].timesEquals(momentum); }
                // apply momentum
                for (int k = 0; k < gradient.length; k++) { this.layers[k].plusEquals(prev_gradient[k]); }
            }
            // update prev_gradient
            prev_gradient = gradient;
            
//            if( (iterations == 0) || (iterations == maximum_iterations / 2) || (iterations == maximum_iterations - 2) ) {
//                    System.out.println("OUTPUT LAYER GRADIENT");
//                    System.out.println(gradient[gradient.length-1]);
//                    System.out.println("OUTPUT LAYER WEIGHTS");
//                    System.out.println(this.layers[layers.length-1].getWeights());
//            }
            
            // Output progress to console and check for convergence
            if(iterations % (maximum_iterations/100) == 0) {
                //System.out.print("-> Training MLP network iteration: " + iterations);
                converged = hasConverged(gradient, true); // Verbose to print status
            } else {
                converged = hasConverged(gradient, false);
            }
            iterations++;
        }
        //System.out.println(layers[0].getWeights());
    }

    @Override
    public double[] test(Set testing_set) {
        ArrayList<Example> examples = testing_set.getExamples(); // Init empty array
        double[] predictions = new double[examples.size()];
        for (int i = 0; i < examples.size(); i++) { // Iterate through all examples
            predictions[i] = predict(examples.get(i));
        }
        return predictions;
    }

    /**
     * Predict the real value or class of an example. This is done by inputing
     * the example, feeding forward, and making a decision from the outputs of
     * the output layer. For classification, find the maximum output and take
     * take that as the predicted class. For regression, take the value of the
     * output.
     *
     * @param ex The example to predict a class or real value of
     * @return
     */
    @Override
    public double predict(Example ex) {
        // Propagate the example through the network and look at the output
        Vector outputs = genLayerOutputs(ex)[layers.length];

        // Check how many nodes are in the output layer to determine 
        // classification or regression.
        if (outputs.getLength() == 1) {
            // The set is regression, so return the one output as the predicted real value.
            return outputs.get(0);
        } else {
            // The set is classification, so return the class of the output node
            // that has the highest activation value.
            return (double) outputs.getMaxIndex();
        }
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
     * method to return the layer at the specified index
     * @param index
     * @return 
     */
    public Layer getLayer(int index) { return this.layers[index]; }

    /**
     * method to initialize layers with correct dimensions and populate weights
     * @param num_classes
     * @param input_dim 
     */
    private void initializeLayers(int num_classes, int input_dim) {
        if (num_hidden_nodes.length == 0) { 
            if (num_classes == -1) { // The set is regression
                // The output layer consists of one node with a linear activation function
                layers[layers.length - 1] = new Layer(new Linear(), 1, input_dim + 1);
            } else { // The set is classification
                // The output layer consists of one node for each class with a sigmoidal activation function
                layers[layers.length - 1] = new Layer(new Logistic(), num_classes, input_dim + 1);
            }
        }
        else { 
            layers[0] = new Layer(new Logistic(), num_hidden_nodes[0], input_dim + 1); 
            for (int i = 1; i < layers.length - 1; i++) {
                layers[i] = new Layer(new Logistic(), num_hidden_nodes[i], layers[i - 1].getNumNodes() + 1);
            }
            // Construct the output layer
            if (num_classes == -1) { // The set is regression
                // The output layer consists of one node with a linear activation function
                layers[layers.length - 1] = new Layer(new Linear(), 1, layers[layers.length - 2].getNumNodes() + 1);
            } else { // The set is classification
                // The output layer consists of one node for each class with a sigmoidal activation function
                layers[layers.length - 1] = new Layer(new Logistic(), num_classes, layers[layers.length - 2].getNumNodes() + 1);
            }
        }
        // Randomly initialize the output layer weights
        for (int i = 0; i < layers.length; i++) {
            layers[i].randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        }
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
    
}
