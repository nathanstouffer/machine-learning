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
import neuralnets.layer.Layer;
import neuralnets.layer.Linear;
import neuralnets.layer.Logistic;
import neuralnets.layer.Matrix;
import neuralnets.layer.Vector;

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
     * Private variable to store the number of weights in the MLP
     * This will be used when converting the network to vector
     * format for ease of use in population based training algorithms
     */
    private int num_weights;
    
    /**
     * storing the similarity matrix for categorical variables
     */
    private final SimilarityMatrix[] sim;

    /**
     * The array of layers holds the weights that make up the network.
     * Feeding an example into the layers produces a prediction
     */
    private Layer[] layers;
    private final int num_hl;
    private final int[] num_hn;

    /**
     * Public constructor to create a new MLP with weights 
     * initialized to 0.
     * Weights can set using public methods in this class
     * @param _num_hidden_layers
     * @param _num_hidden_nodes
     * @param _output_dim
     * @param _input_dim
     * @param _sim 
     */
    public MLP(int _num_hidden_layers, int[] _num_hidden_nodes, 
            int _output_dim, int _input_dim, SimilarityMatrix[] _sim) {
        this.num_hl = _num_hidden_layers;
        this.num_hn = new int[num_hl];
        this.sim = _sim;
        // randomly populate layers
        for (int i = 0; i < num_hn.length; i++) { num_hn[i] = _num_hidden_nodes[i]; }
        this.layers = new Layer[num_hl + 1];
        this.initializeLayers(_input_dim, _output_dim);
        // set global variable for number of weights
        this.num_weights = 0;
        for (int l = 0; l < this.layers.length; l++) { 
            this.num_weights += this.layers[l].getNumWeights(); 
        }
    }
    
    /**
     * constructor to create a new MLP with all weights initialized to 0.
     * Weights can be set using public methods randPopWeights() and 
     * setWeights(Vector vec)
     * 
     * the array topology is expected to have the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth 
     * hidden layer
     * 
     * @param topology of the form: { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * @param _sim 
     */
    public MLP(int[] topology, SimilarityMatrix[] _sim) {
        // set global variables
        this.sim = _sim;
        this.num_hl = topology[0];
        this.num_hn = new int[this.num_hl];
        for (int h = 0; h < this.num_hn.length; h++) { this.num_hn[h] = topology[h+1]; }
        // set layers
        this.layers = new Layer[num_hl + 1];
        int len = topology.length;
        this.initializeLayers(topology[len-2], topology[len-1]);
        // set global variable for number of weights
        this.num_weights = 0;
        for (int l = 0; l < this.layers.length; l++) { 
            this.num_weights += this.layers[l].getNumWeights(); 
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
     * method to return the layer at the specified index
     * @param index
     * @return 
     */
    public Layer getLayer(int index) { return this.layers[index]; }
    
    /**
     * public method to convert the layers of a network to vector
     * format for ease of manipulation with population based algorithms
     * @return 
     */
    public Vector toVec() {
        // initialize Vector
        Vector vec = new Vector(this.num_weights);
        
        // we now populate the vector
        int index = 0;
        // iterate through layers
        for (int l = 0; l < this.layers.length; l++) {
            Matrix weights = this.layers[l].getWeights();
            // iterate through matrix
            for (int r = 0; r < weights.getNumRows(); r++) {
                Vector row = weights.getRow(r);
                // iterate through vector
                for (int v = 0; v < row.getLength(); v++) {
                    vec.set(index, row.get(v));
                    index++;
                }
            }
        }
        
        // return vector
        return vec;
    }
    
    /**
     * public method to set the vec of the network
     * according to the vec in the parameter
     * 
     * It is assumed that the vec have a one-to-one
     * correspondence with the vec returned by the 
     * public method toVec()
     * 
     * @param vec 
     */
    public void setWeights(Vector vec) {
        if (vec.getLength() != this.num_weights) { System.err.println("weights arg is not compatible with network"); }
        else {
            // network and vector are compatible
            int v = 0;      // current index of vec
            // iterate through layers
            for (int l = 0; l < this.layers.length; l++) {
                int num_rows = this.layers[l].getNumNodes();
                int num_col = this.layers[l].getNumInputs();
                
                // instantiate new matrix
                Matrix mtx = new Matrix(num_rows, num_col);
                // populate rows of matrix
                for (int m = 0; m < mtx.getNumRows(); m++) {
                    // this will be a row in the matrix
                    Vector temp = new Vector(num_col);
                    // populate temp
                    for (int t = 0; t < temp.getLength(); t++) {
                        temp.set(t, vec.get(v));
                        v++;        // increment v
                    }
                    mtx.setRow(m, temp);
                }
                
                // set weights
                this.layers[l].setWeights(mtx);
            }
        }
    }
    
    /**
     * public method to randomly populate all weights in a network. 
     * The values are bounded by STARTING_WEIGHT_BOUND
     */
    public void randPopWeights() {
        // Randomly initialize the output layer weights
        for (int i = 0; i < layers.length; i++) {
            layers[i].randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        }
    }

    /**
     * method to initialize layers with correct dimensions
     * @param output_dim
     * @param input_dim 
     */
    private void initializeLayers(int input_dim, int output_dim) {
        if (num_hn.length == 0) { 
            if (output_dim == -1) { // The set is regression
                // The output layer consists of one node with a linear activation function
                layers[layers.length - 1] = new Layer(new Linear(), 1, input_dim + 1);
            } else { // The set is classification
                // The output layer consists of one node for each class with a sigmoidal activation function
                layers[layers.length - 1] = new Layer(new Logistic(), output_dim, input_dim + 1);
            }
        }
        else { 
            layers[0] = new Layer(new Logistic(), num_hn[0], input_dim + 1); 
            for (int i = 1; i < layers.length - 1; i++) {
                layers[i] = new Layer(new Logistic(), num_hn[i], layers[i - 1].getNumNodes() + 1);
            }
            // Construct the output layer
            if (output_dim == -1) { // The set is regression
                // The output layer consists of one node with a linear activation function
                layers[layers.length - 1] = new Layer(new Linear(), 1, layers[layers.length - 2].getNumNodes() + 1);
            } else { // The set is classification
                // The output layer consists of one node for each class with a sigmoidal activation function
                layers[layers.length - 1] = new Layer(new Logistic(), output_dim, layers[layers.length - 2].getNumNodes() + 1);
            }
        }
    }
    
}
