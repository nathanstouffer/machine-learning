package nearestneighboralgorithm;

import java.util.Iterator;

/**
 *  K-NN Classifier
 * 
 * @author andy-
 * 
 * The K-NN classifier is a K-Nearest Neighbor algorithm implementation that
 * works with classification data sets.
 * Given a set of examples, any new examples will be classified by a popular 
 * vote amongst its k nearest neighbors.
 */
public class KNNClassifier implements IKNearestNeighbor {

    private IDistMetric dist_metric; // The metric that will be used to measure
                                     // distance between examples
    
    private Set neighbors; // The set of examples that the K-NN will come from
    
    private int k; // The number of nearest neigbors to check.
    
    /**
     * Creates a K-NN Classifier
     * (Empty Constructor)
     */
    public KNNClassifier() {
    }
    
    /**
     * Set the distance metric used by the classifier.
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
     * Trains the classifier with a given set.
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
     * @param testing_set The set to classify.
     * @return double[] The predicted classifications of the testing set, in the
     * same order as the set provided.
     */
    @Override
    public double[] test(Set testing_set) {
        double[] classifications = new double[testing_set.getNumExamples()];
        // Iterate through all examples, classifying them
        Iterator<Example> iterator = testing_set.iterator();
        int i = 0; //Track which example is being classified
        while(iterator.hasNext()) {
            classifications[i] = classify(iterator.next());
        }
        return classifications;
    }
    
    /**
     * Classify an example by determining its nearest neighbors and choosing the
     * most frequent class among them.
     * @param example The example to classify.
     * @return double The predicted class that the example belongs to.
     */
    private double classify(Example example) {
        // Initialize two arrays to track the nearest neighbors. Only the class
        // and distance of the k-nn are necessary to hold on to.
        double[] nn_classes = new double[k];
        double[] nn_distances = new double[k];
        for(int i = 0; i < k; i++) { 
            //Initialize the values so they will be overwritten
            nn_classes[i] = -1;
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
                    nn_classes[i] = neighbor.getValue();            //THIS (.getClassType()  SHOULD BE MODIFIED IN THE FUTURE SO IT SUPPORTS DOUBLE VALUESSSSSSSS
                    nn_distances[i] = dist;
                    break; // Exit for-loop
                } //Otherwise, check next k-nn
            }
        }
        
        // Now that the k-nn have been found, find the most frequent class
        // among them.
        int num_classes = neighbors.getNumClasses();
        int[] class_freq = new int[num_classes]; // Create a histogram with each class
        for(int i = 0; i < k; i++) { // Populate the histogram
            if(nn_classes[i] != -1) { // Make sure there were an appropriate # of nn
                class_freq[(int)nn_classes[i]]++;
            }
        }
        int classification = 0;
        for(int i = 0; i < num_classes; i++) { // Find the most frequent in the histogram
            if(class_freq[i] > class_freq[classification]) {
                classification = i;
            }
        }
        
        return classification;
    }
    
}
