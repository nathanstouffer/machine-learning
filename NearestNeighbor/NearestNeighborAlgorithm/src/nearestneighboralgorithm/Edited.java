/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 * class to reduce the size of a data set for the use of 
 * a Nearest Neighbor algorithm.
 * 
 * The algorithm (Edited Nearest Neighbor) edits examples
 * out of the data set that the surrounding k-nearest examples
 * misclassify
 * 
 * @author natha
 */
public class Edited implements IDataReducer {
    
    // variable to determine how many neighbors the reducing
    // algorithm should view
    private int k;
    // metric that the reduciing algorithm should use
    private IDistMetric metric;
    // set that exists for validation purposes to test
    // whether the reducing algorithm should be terminated
    private Set validation_set;    
    
    /**
     * constructor to initialize global variables
     * @param k
     * @param metric
     * @param validation_set 
     */
    Edited(int k, IDistMetric metric, Set validation_set){
        this.k = k;
        this.metric = metric;
        this.validation_set = validation_set;
    }
    
    /**
     * method to recursively reduce the size of a set for
     * the use of k-nearest neighbors as long as performance
     * does not degrade. This will be determined using
     * the accuracy metric since Edited Nearest Neighbors is
     * used only on classification problems
     * @param orig
     * @return 
     */
    public Set reduce(Set orig){
        // assume that 
    }
    
    private Set reduce_rec(Set orig, double orig_accuracy){
        // initialize a learner with the original set
        Classifier orig_learner = new Classifier(this.k, orig, this.metric); // TODO: edit meet Andy's constructor
        
        // using orig_learner, classify all points in orig
        for (Example ex: orig){
            // remove point if missclassified
        }
        
        // initialize a new learner
        
        // evaluate both learners with validation set
        
        // compare accuracies and act accordingly
    }
    
}
