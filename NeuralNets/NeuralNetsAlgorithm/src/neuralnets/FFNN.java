/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
import measuredistance.IDistMetric;
import networklayer.IActFunct;
import networklayer.Layer;
import networklayer.Linear;
import networklayer.Logistic;
import networklayer.Vector;

/**
 *
 * @author Kevin
 */
public class FFNN implements INeuralNet{
    
    private static final double STARTING_WEIGHT_BOUND = 0.01;
    
    private final double learning_rate;
    private final double batch_size;
    
    private Layer output_layer;
    private Layer[] hidden_layer;
    /**
     * activation functions for each hidden layer, each index corresponds
     * with a hidden layer layer
     */
    private IActFunct[] activation_function;
    
    private final int num_hidden_layer;
    private final int num_hidden_nodes;
    
    private final IDistMetric dist_metric;
    FFNN(int _num_hidden_layers, int _num_hidden_nodes, double _learning_rate, double _batch_size, IDistMetric _dist_metric, IActFunct[] _activation_function){
        num_hidden_layer = _num_hidden_layers;
        num_hidden_nodes = _num_hidden_nodes;
        learning_rate = _learning_rate;
        batch_size = _batch_size;
        dist_metric = _dist_metric; 
        activation_function = _activation_function;
    }
    
    @Override
    public void train(Set training_set){
        // Construct the output layer
        if(training_set.getNumClasses() == -1) { // The set is regression
            // The output layer consists of one node with a linear activation function
            output_layer = new Layer(new Linear(), 1, training_set.getNumExamples());
        } else { // The set is classification
            // The output layer consists of one node for each class with a sigmoidal activation function
            output_layer = new Layer(new Logistic(), training_set.getNumClasses(), training_set.getNumExamples());
        }
        for(int i = 0; i < num_hidden_layer; i++){
            hidden_layer[i] = new Layer(activation_function[i], num_hidden_nodes, training_set.getNumExamples());
        }
        // Randomly initialize the output layer weights
        output_layer.randPopulate(-STARTING_WEIGHT_BOUND, STARTING_WEIGHT_BOUND);
        
        // Create a backpropagator that will just train the output layer.
        Backpropagator backprop = new Backpropagator(this);
    }
    @Override
    public double[] test (Set testing_set){
        
    }
    @Override
    public double predict(Example ex){
        
    }
    @Override
    public Vector[] genLayerOutputs(Example ex){
        
    }
    @Override
    public Vector[] genLayerDeriv(){
        
    }
    @Override
    public int[][] getLayerDim(){
        
    } 
    
}
