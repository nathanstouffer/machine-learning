package nearestneighboralgorithm;

import java.util.Iterator;

/**
 * K-NN with Regression
 * @author andy-
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
        Iterator<Example> iterator = testing_set.iterator();
        int i = 0; //Track which example is being handled
        while(iterator.hasNext()) {
            predictions[i] = predict(iterator.next());
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
        double[] nn_values = new double[k];
        double[] nn_distances = new double[k];
        for(int i = 0; i < k; i++) { 
            //Initialize the values so they will be overwritten
            nn_values[i] = 0;
            nn_distances[i] = Double.MAX_VALUE;
        }
        
        // Iterate through all neighbors, calculating their distance from the
        // example and comparing to the current k-nn.
        Iterator<Example> iterator = neighbors.iterator();
        while(iterator.hasNext()) {
            // Calculate the distance
            Example neighbor = iterator.next();
            double dist = dist_metric.dist(example, neighbor);
            // Check if the distance is smaller than any current k-nn
            for(int i = 0; i < k; i++) {
                if(dist < nn_distances[i]) {
                    // If closer, then overwrite the k-nn
                    nn_values[i] = neighbor.getClassType();            //THIS (.getClassType()  SHOULD BE MODIFIED IN THE FUTURE SO IT SUPPORTS DOUBLE VALUESSSSSSSS
                    nn_distances[i] = dist;
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
            if(nn_distances[i] < Double.MIN_VALUE) { // Make sure it that there were an
                                                     // appropriate # of nn
                sum += nn_values[i];
                num++;
            } else { // Otherwise, compute the mean based on the nn that exist
                break;
            }
        }
        return sum / num;
    }
    
}
