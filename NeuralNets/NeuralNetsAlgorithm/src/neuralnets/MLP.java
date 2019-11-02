/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
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

    private static final double STARTING_WEIGHT_BOUND = 0.01;

    private final double learning_rate;
    private final double batch_size;

    private Layer[] layers;

    private final double momentum;

    private final int num_hidden_layer;
    private final int num_hidden_nodes;

    MLP(int _num_hidden_layers, int _num_hidden_nodes, double _learning_rate, double _batch_size, double _momentum) {
        num_hidden_layer = _num_hidden_layers;
        num_hidden_nodes = _num_hidden_nodes;
        learning_rate = _learning_rate;
        batch_size = _batch_size;
        momentum = _momentum;
        layers = new Layer[num_hidden_layer + 1];

    }

    @Override
    public void train(Set training_set) {
        layers[0] = new Layer(new Logistic(), num_hidden_nodes, training_set.getNumAttributes() + 1);
        for (int i = 1; i < layers.length - 1; i++) {
            layers[i] = new Layer(new Logistic(), num_hidden_nodes, layers[i].getNumNodes() + 1);
        }
        // Construct the output layer
        if (training_set.getNumClasses() == -1) { // The set is regression
            // The output layer consists of one node with a linear activation function
            layers[layers.length - 1] = new Layer(new Linear(), 1, layers[layers.length - 2].getNumNodes());
        } else { // The set is classification
            // The output layer consists of one node for each class with a sigmoidal activation function
            layers[layers.length - 1] = new Layer(new Logistic(), training_set.getNumClasses(), layers[layers.length - 2].getNumNodes());
        }
        // Randomly initialize the output layer weights
        for (int i = 0; i < layers.length; i++) {
            layers[i].randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        }

        // Create a backpropagator that will just train the output layer.
        Backpropagator backprop = new Backpropagator(this);

        boolean converged = false;
        int iterations = 0;
        while (!converged) {
            // Output progress to console
            if (iterations % 100 == 0) {
                System.out.println("-> Training RBF network iteration: " + iterations);
            }
            // Send each batch through the output layer
            Matrix[] prev_gradient = null;
            Set[] batches = training_set.getRandomBatches(batch_size);
            for (int j = 0; j < batches.length; j++) {
                // Get gradient
                Matrix[] gradient = backprop.computeGradient(batches[j]);
                // Multiply gradient with learning rate
                
                for (int k = 0; k < gradient.length; k++) { gradient[k].timesEquals(learning_rate); }
                // apply momentum if necessary
                if (momentum != 0.0 && prev_gradient == null) {
                    // multiply by momentum rate
                    for (int k = 0; k < prev_gradient.length; k++) {
                        prev_gradient[k].timesEquals(momentum);
                    }
                    // update gradient
                    for (int k = 0; k < gradient.length; k++) {
                        gradient[k].plusEquals(prev_gradient[k]);
                    }
                }
                // update weights
                for (int k = 0; k < layers.length; k++) {
                    layers[k].plusEquals(gradient[k]);
                }
            }
            iterations++;
            if (iterations == 500) {
                converged = true;
            }
        }
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
        Vector[] outputs = genLayerOutputs(ex);

        // Check how many nodes are in the output layer to determine 
        // classification or regression.
        if (outputs[layers.length - 1].getLength() == 1) {
            // The set is regression, so return the one output as the predicted real value.
            return outputs[layers.length - 1].get(0);
        } else {
            // The set is classification, so return the class of the output node
            // that has the highest activation value.
            return (double) outputs[layers.length - 1].getMaxIndex();
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
        outputs[0] = new Vector(ex);
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

}
