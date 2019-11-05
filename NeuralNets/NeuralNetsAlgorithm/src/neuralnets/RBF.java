package neuralnets;

import datastorage.Example;
import datastorage.Set;
import java.util.ArrayList;
import measuredistance.IDistMetric;
import networklayer.Layer;
import networklayer.Linear;
import networklayer.Logistic;
import networklayer.Matrix;
import networklayer.Vector;

/**
 *
 * @author andy-
 */
public class RBF implements INeuralNet {

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
     * termination characteristics during training. Training the output layer
     * weights will terminate when the gradient updates are all weighted less
     * than the convergence threshold (as a percentage) multiplied by the
     * current weights or when the number of training iterations reaches the
     * specified maximum.
     */
    private final double convergence_threshold;
    private final int maximum_iterations;

    /**
     * The RBFs need a way to compute distance from the representative example
     * of the node. All RBFs will use the supplied distance metric dist_metric.
     */
    private final IDistMetric dist_metric;

    /**
     * Private variables representatives and variances are the examples and
     * associated variances that represent each node in the hidden layer and its
     * associated Gaussian radial basis function.
     */
    private final Set representatives;
    private final double variances[];

    /**
     * The output layer weights are contained in output_layer. The output_layer
     * is initialized during training.
     */
    private Layer output_layer;

    /**
     * Create a radial basis function neural network given a set of
     * representative examples and corresponding variances.
     * @param _representatives The set of examples to be used in the
     * construction of the RBFs.
     * @param _variances The variances to be used in the construction of the
     * RBFs. Must be in the same order as the representatives set.
     * @param _learning_rate The learning rate of the network, affecting how
     * much of an impact each weight update will have and the speed at which the
     * network will learn.
     * @param _batch_size The percentage of examples in each training set to use
     * at a time when applying gradient descent to the output layer.
     * @param _convergence_threshold The threshold to determine when to stop
     * training the output layer.
     * @param _maximum_iterations The maximum iterations allowed during training
     * of the output layer.
     * @param _dist_metric The distance metric that will be used in the Gaussian
     * radial basis function. Euclidean distance will be the most common.
     */
    public RBF (Set _representatives, double[] _variances,
                double _learning_rate, double _batch_size, 
                double _convergence_threshold, int _maximum_iterations,
                IDistMetric _dist_metric) {
        representatives = _representatives;
        variances = _variances;
        learning_rate = _learning_rate;
        batch_size = _batch_size;
        convergence_threshold = _convergence_threshold;
        maximum_iterations = _maximum_iterations;
        dist_metric = _dist_metric;
    }

    /**
     * Trains the RBF network. Applies gradient descent at the output layer
     * using the examples in the training set. The training set will be broken
     * into batches (output layer updates are the sum of the gradient results of
     * each example in the batch). The output layer (what we are training) is
     * only updated at the completion of each batch.
     *
     * Upon training, a regression or classification set will be automatically
     * detected and the output layer constructed accordingly.
     * @param training_set The set to train the RBF network with.
     */
    @Override
    public void train(Set training_set) {
        // Construct the output layer
        if(training_set.getNumClasses() == -1) { // The set is regression
            // The output layer consists of one node with a linear activation function
            output_layer = new Layer(new Linear(), 1, representatives.getNumExamples() + 1);
        } else { // The set is classification
            // The output layer consists of one node for each class with a sigmoidal activation function
            output_layer = new Layer(new Logistic(), training_set.getNumClasses(), representatives.getNumExamples() + 1);
        }
        // Randomly initialize the output layer weights
        output_layer.randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);

        // Create a backpropagator that will just train the output layer.
        Backpropagator backprop = new Backpropagator(this);

//        Set[] batches = training_set.getRandomBatches(batch_size);        //TEST IF YOU WANT IT TO RUN about 10% FASTER

        boolean converged = false;
        int iterations = 0;
        while(!converged && iterations < maximum_iterations) {
            // Send the first batch through
            Set batch = training_set.getRandomBatch(batch_size);

            // Get gradient
            Matrix gradient = backprop.computeGradient(batch)[0];
            // Multiply gradient with learning rate
            gradient.timesEquals(learning_rate);
            // Apply gradient to output layer
            output_layer.plusEquals(gradient);

            if(( iterations == 0) || (iterations == maximum_iterations / 2) || (iterations == maximum_iterations - 2) ) {
                    System.out.println("GRADIENT");
                    System.out.println(gradient);
                    System.out.println("OUTPUT LAYER WEIGHTS");
                    System.out.println(output_layer.getWeights());
            }

            // Output progress to console and check for convergence
            if(iterations % (maximum_iterations/100) == 0) {
                System.out.print("-> Training RBF network iteration: " + iterations);
                converged = hasConverged(gradient, true); // Verbose to print status
            } else {
                converged = hasConverged(gradient, false);
            }
            iterations++;
        }
        System.out.println("-> Trained RBF on iteration: " + iterations);
        // Output layer weights are done training!
    }

    /**
     * Tests a set, predicting classification or real value.
     * @param testing_set The set to test.
     * @return double[] The predicted classes or real values, in the same order
     * as the test set passed in.
     */
    @Override
    public double[] test(Set testing_set) {
        ArrayList<Example> examples = testing_set.getExamples(); // Init empty array
        double[] predictions = new double[examples.size()];
        for(int i = 0; i < examples.size(); i++) { // Iterate through all examples
            predictions[i] = predict(examples.get(i));
        }
        return predictions;
    }

    /**
     * Predict the real value or class of an example. This is done by inputing
     * the example, feeding forward, and making a decision from the outputs
     * of the output layer. For classification, find the maximum output and take
     * take that as the predicted class. For regression, take the value of the
     * output.
     * @param ex The example to predict a class or real value of
     * @return
     */
    @Override
    public double predict(Example ex) {
        // Propagate the example through the network and look at the output
        Vector outputs = genLayerOutputs(ex)[1];

//        System.out.println("WEIGHTS " + output_layer.getWeights());
//        System.out.println("HIDDEN ACTS " + genLayerOutputs(ex)[0]);
//        System.out.println("OUTPUTS: " + outputs);

        // Check how many nodes are in the output layer to determine
        // classification or regression.
        if(outputs.getLength() == 1) {
            // The set is regression, so return the one output as the predicted real value.
            //System.out.println("PREDICTED VALUE: " + outputs.get(0));
            return outputs.get(0);
        } else {
            // The set is classification, so return the class of the output node
            // that has the highest activation value.
            //System.out.println("PREDICTED CLASS: " + (double)outputs.getMaxIndex());
            return (double)outputs.getMaxIndex();
        }
    }

    /**
     * Returns the output of each layer, in this case the output of the RBF
     * layer and the output layer (in that order).
     * @param ex
     * @return A vector array containing the output of each layer.
     */
    @Override
    public Vector[] genLayerOutputs(Example ex) {
        //Initialize vector array
        Vector[] outputs = new Vector[2];

        // Initialize vector for RBF layer
        Vector RBF_outputs = new Vector(representatives.getNumExamples());
        // Calculate each RBF node's output given the example as input
        for(int i = 0; i < representatives.getNumExamples(); i++) { 
            // Output is the calculated using the radial basis function:
            //          o = exp(-(x1 - x2)^2 / (2*variance))
            double d = dist_metric.dist(representatives.getExample(i), ex);
            double o = Math.exp( -(d) / (2 * variances[i]) );
            RBF_outputs.set(i, o);
        }
        // insert multiplier for the weight
        RBF_outputs.insertBiasMultiplier();
        outputs[0] = RBF_outputs;

        outputs[1] = output_layer.feedForward(RBF_outputs);

        return outputs;
    }

    /**
     * Returns the derivative of each layer, in this case, just the output
     * layer.
     * Note: genLayerOutputs() must be called prior to this function call, or
     * there will be no current derivative.
     * @return
     */
    @Override
    public Vector[] genLayerDeriv() {
        Vector[] deriv = new Vector[1];
        deriv[0] = output_layer.getDeriv();
        return deriv;
    }

    /**
     * Returns the dimensions of the layers.
     * @return A two dimensional int array. First index is layer. Second index
     * is: 0 -> # nodes and 1 -> # inputs to the layer.
     */
    @Override
    public int[][] getLayerDim() {
        // Backprop will only be using the output layer.
        int[][] dim = new int[1][2];
        dim[0][0] = output_layer.getNumNodes();
        dim[0][1] = output_layer.getNumInputs();
        return dim;
    }

    /**
     * Check if the network has converged. Will return true if no values within
     * the gradient are higher than CONVERGENCE_THRESHOLD (a percentage) of the
     * existing layer weights.
     * @return
     */
    private boolean hasConverged(Matrix gradient, boolean verbose) {
        double avg = 0;
        Matrix weights = output_layer.getWeights();
        // Iterate through rows
        for(int row = 0; row < weights.getNumRows(); row++) {
            Vector grad_row = gradient.getRow(row);
            Vector weight_row = weights.getRow(row);
            // Iterate through cols
            for(int col = 0; col < weights.getNumCol(); col++) {
                double g = Math.abs(grad_row.get(col));
                double w = Math.abs(weight_row.get(col));
                // Test if the value exceeds the threshol
                if( g > w*convergence_threshold) {
                    if(verbose) {
                        System.out.print("    ---> GRADIENT TO WEIGHT RATIO = " + (g/w) + " WHERE THRESHOLD = " + convergence_threshold);
                        System.out.println("    ---> W = " + w + " G = " + g);
                    }
                    return false;
                }
            }
        }
        return true;
    }
    
    /**
     * method to return a layer
     * @param index useless parameter to satisfy interface
     * @return 
     */
    public Layer getLayer(int index) { return this.output_layer; }

}
