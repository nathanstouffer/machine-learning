package knearestneighbor;

import datastorage.Example;
import datastorage.Set;
import measuredistance.IDistMetric;
import java.util.ArrayList;

/**
 * K-NN with Regression
 * @author andy-
 * 
 * The K-NN regressor is a K-Nearest Neighbor algorithm implementation that
 * works with regression data sets.
 * Given a set of examples, any new examples' values will be predicted by the  
 * mean value of its k nearest neighbors.
 */
public class KNNRegressor implements IKNearestNeighbor {

    private IDistMetric dist_metric; // The metric that will be used to measure
                                     // distance between examples
    
    private Set neighbors; // The set of examples that the K-NN will come from
    
    private int k; // The number of nearest neigbors to check.
    
    /**
     * Creates a K-NN Regressor
     * (Empty Constructor)
     */
    public KNNRegressor() {
    }
    
    /**
     * Set the distance metric used by the regressor.
     * @param metric The distance metric to be used.
     */
    @Override
    public void setDistMetric(IDistMetric metric) {
        dist_metric = metric;
    }
    
    /**
     * Sets k, the number of nearest neighbors to check
     * @param new_k The new value of k.
     */
    @Override
    public void setK(int new_k) {
        k = new_k;
    }

    /**
     * Trains the regressor with a given set.
     * Training merely consists of saving the set
     * for later use.
     * @param training_set 
     */
    @Override
    public void train(Set training_set) {
        neighbors = training_set;
    }

    /**
     * Tests a set of examples.
     * @param testing_set The set to predict real values of.
     * @return double[] The predicted values of the testing set, in the
     * same order as the set provided.
     */
    @Override
    public double[] test(Set testing_set) {
        double[] predictions = new double[testing_set.getNumExamples()];
        // Iterate through all examples, predicting their real values
        //Iterator<Example> iterator = testing_set.iterator();
        //int i = 0; //Track which example is being handled
        //while(iterator.hasNext()) {
            //predictions[i] = predict(iterator.next());
        for (int i = 0; i < testing_set.getNumExamples(); i++) {
            predictions[i] = predict(testing_set.getExample(i));
        }
        return predictions;
    }
    
    /**
     * Predict the real value of an example by determining its nearest 
     * neighbors and choosing the mean value among them.
     * @param example The example to predict a real value for.
     * @return double The predicted value of the example.
     */
    private double predict(Example example) {
        // Initialize two arrays to track the nearest neighbors. Only the value
        // and distance of the k-nn are necessary to hold on to.
        ArrayList<Double> nn_values = new ArrayList<>();
        ArrayList<Double> nn_distances = new ArrayList<>();
        for(int i = 0; i < k; i++) { 
            //Initialize the values so they will be overwritten
            nn_values.add(0.0);
            nn_distances.add(Double.MAX_VALUE);
        }
        
        // Iterate through all neighbors, calculating their distance from the
        // example and comparing to the current k-nn.
        for (int a = 0; a < neighbors.getNumExamples(); a++) {
            // Calculate the distance
            Example neighbor = neighbors.getExample(a);
            double dist = dist_metric.dist(example, neighbor);
            // Check if the distance is smaller than any current k-nn
            for(int i = 0; i < k; i++) {
                if(dist < nn_distances.get(i)) {
                    // If closer, then overwrite the k-nn
                    nn_values.add(i, neighbor.getValue());
                    nn_distances.add(i, dist);
                    // Delete last k-nn
                    nn_values.remove(k);
                    nn_distances.remove(k);
                    break; // Exit for-loop
                } //Otherwise, check next k-nn
            }
        }
        
        // Now that the k-nn have been found, find mean value of them and
        // return that as the predicted value
        double sum = 0;
        double num = 0; // The number of nearest neigbors (may be less than k if
                        // there are less than k in the data set).
        for(int i = 0; i < k; i++) {
            if(nn_distances.get(i) < Double.MAX_VALUE) { // Make sure it that there were an
                                                     // appropriate # of nn
                sum += nn_values.get(i);
                num++;
            } else { // Otherwise, compute the mean based on the nn that exist
                break;
            }
        }
        
        //System.out.println("Prediction: " + sum/num);
        //System.out.println("Real value: " + example.getValue());
        return sum / num;
    }
    
}
