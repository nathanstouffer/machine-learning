package nearestneighboralgorithm;

/**
 *  K-NN Classifier
 * 
 * @author andy-
 * 
 * The K-NN classifier is a K-Nearest Neighbor algorithm implementation that
 * works with classification data sets.
 * Given a set of examples as a model, any new examples will be classified 
 */
public class KNNClassifier implements IKNearestNeighbor {

    private IMetric dist_metric; // The metric that will be used to measure
                                 // distance between examples
    
    private Set model; // The set of examples that the K-NN will come from
    
    private int k; // The number of nearest neigbors to check.
    
    /**
     * Creates a K-NN Classifier with a given k.
     * @param _k The number of nearest neighbors to check
     */
    public KNNClassifier(int _k) {
        k = _k;
    }
    
    @Override
    public void setDistMetric(IMetric metric) {
        dist_metric = metric;
    }

    /**
     * Trains the classifier with a given set.
     * Training merely consists of saving the set
     * for later use.
     * @param training_set 
     */
    @Override
    public void train(Set training_set) {
        model = training_set;
    }

    /**
     * Tests/Classifies a set of examples.
     * @param testing_set
     * @return double[] The predicted classifications of the testing set, in the
     * same order as the set provided.
     */
    @Override
    public double[] test(Set testing_set) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /**
     * @param example The example to classify
     * @return double The predicted class that the example belongs to.
     */
    private double classify(Example example) {
        return 0.0;
    }
    
}
