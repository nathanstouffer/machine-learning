package nearestneighboralgorithm;

/**
 * IKNearestNeigbor Interface
 * @author andy-
 * 
 * Defines the functions inherent to a K-NN learning algorithm
 */
public interface IKNearestNeighbor {
    
    public void setDistMetric(IDistMetric metric);
    
    public void setK(int k);
    
    public void train(Set training_set);
    
    public double[] test(Set testing_set);
}
