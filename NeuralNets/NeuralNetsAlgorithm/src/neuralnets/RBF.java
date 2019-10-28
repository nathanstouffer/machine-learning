package neuralnets;

import datastorage.Example;
import datastorage.Set;
import java.util.ArrayList;
import measuredistance.IDistMetric;
import networklayer.Layer;
import networklayer.Linear;
import networklayer.Logistic;
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
    private static final double STARTING_WEIGHT_BOUND = 0.01;
    
    
    /**
     * The learning rate for the network is a tunable parameter that affects
     * the impact of each iteration of weight updates on the output layer.
     * Batch size is the percentage of examples in each training set to use
     * at one time while applying gradient descent to the output layer.
     */
    private final double learning_rate;
    private final double batch_size;
    
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
     * is initialized during training. Although there is only one layer here, it
     * is stored as an array for ease of use with the backpropagator class.
     */
    private Layer[] output_layer;
    
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
     * @param _dist_metric The distance metric that will be used in the Gaussian 
     * radial basis function. Euclidean distance will be the most common.
     */
    public RBF (Set _representatives, double[] _variances, double _learning_rate, double _batch_size, IDistMetric _dist_metric) {
        representatives = _representatives;
        variances = _variances;
        learning_rate = _learning_rate;
        batch_size = _batch_size;
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
        output_layer = new Layer[1];
        if(training_set.getNumClasses() == -1) { // The set is regression
            // The output layer consists of one node with a linear activation function
            output_layer[1] = new Layer(new Linear(), 1, representatives.getNumExamples());
        } else { // The set is classification
            // The output layer consists of one node for each class with a sigmoidal activation function
            output_layer[1] = new Layer(new Logistic(), training_set.getNumClasses(), representatives.getNumExamples());
        }
        // Randomly initialize the output layer weights
        output_layer[1].randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        
        // Save the aggregated gradient that will be applied at the end of each batch
        // Layer gradient = new Layer(); SOMETHING LIKE THIS
        for(int i = 0; i < training_set.getNumExamples(); i++) { // Iterate through all examples
            // Call the backpropagator class to apply gradient descent to the output layer
            // GRADIENT UPDATE += RETURNED GRADIENT FROM BACKPROP-
            // Check if the current example is the last in a batch or the last example to train with
            if( ((i+1) % ((int)(training_set.getNumExamples() * batch_size)) == 0)
                    || (i == training_set.getNumExamples())) {
                // Apply the aggregated gradient change
                // output_layer[1]  APPLY CHANGES
            }
        }
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
        
        // Check how many nodes are in the output layer to determine 
        // classification or regression.
        if(outputs.getLength() == 1) {
            // The set is regression, so return the one output as the predicted real value.
            return outputs.get(0);
        } else {
            // The set is classification, so return the class of the output node
            // that has the highest activation value.
            return (double)outputs.getMaxIndex();
        }
    }

    /**
     * Returns the output of each layer, in this case the output of the RBF
     * layer and the output layer (in that order).
     * @param ex
     * @return 
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
        outputs[0] = RBF_outputs;
        
        outputs[1] = output_layer[1].feedForward(RBF_outputs);
        
        return outputs;
    }
    
}
