/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 *
 * @author erick
 */
public class Condensed {
    
    // variable to determine how many neighbors the learning
    // algorithm will view
    private int k;
    // metric that the reduciing algorithm will use
    private IDistMetric metric;
    // learner used for reducing purposes
    // learner must be of type KNNClassifier since Condensed 
    // Nearest Neighbor is limited to classification problems
    private KNNClassifier learner;
    // set that exists for validation purposes to test
    // whether E-NN algorithm should be terminated
    private Set validation_set;    
    
    Condensed(int k, IDistMetric metric, Set validation_set){
        this.k = k;
        this.metric = metric;
        this.learner = new KNNClassifier();
        this.learner.setK(k);
        this.learner.setDistMetric(metric);
        
        this.validation_set = validation_set;
    }
    
    public Set reduce(Set original){
        
        Set preCondense = original.clone();
        Set postCondense = original.clone();
        while(preCondense != postCondense){
            for (Example ex : preCondense){
                metric.dist();
            }
        }
    }
    
}
