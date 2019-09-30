/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList;

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
    // learner used for reducing puruposes
    private KNNClassifier learner;
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
        learner = new KNNClassifier();
        learner.setK(k);
        learner.setDistMetric(metric);
        
        this.validation_set = validation_set;
    }
    
    /**
     * method to reduce the size of a set for the use of k-nearest
     * neighbors as long as performance does not degrade.
     * This will be determined using
     * the accuracy metric since Edited Nearest Neighbors is
     * used only on classification problems
     * @param orig
     * @return 
     */
    public Set reduce(Set orig){
        // clone porig
        Set clone = orig.clone();
        
        // train learner with clone
        learner.train(clone);
        
        boolean edit = true;
        do {
            // compute accuracy for the original learner using validation_set
            double orig_acc = computeAccuracy();
            
            // get misclassifed examples in the data set
            ArrayList<Example> misclassified = findMisclassified(clone);
            
            // delete missclassified examples from orig
            for (Example ex: misclassified){ clone.delExample(ex); }
            
            // learner accesses orig in memory
            // so the data in learner has been edited
            
            // compute accuracy for edited learner using validation_set
            double edited_acc = computeAccuracy();
            
            // stop editing set when performance degrades
            if (edited_acc < orig_acc){ edit = false; }
        } while (edit);
        
        return clone;
    }
    
    /**
     * method to compute the accuracy of a learner using the
     * global variable validation_set
     * @param learner
     * @return 
     */
    private double computeAccuracy(){
        double [] pred = learner.test(validation_set);
        
        // TODO: fix once EvaluateExperiment exists
        return 0.0;
    }
    
    /**
     * method to find the misclassified examples using k-NN
     * @param orig
     * @return 
     */
    private ArrayList<Example> findMisclassified(Set orig){
        ArrayList<Example> misclassified = new ArrayList<Example>();
        // using learner, classify all points in orig
        for (Example ex: orig){
            double real = ex.getValue();
            double pred = learner.classify(ex);

            // if incorrect classification, add ex to misclassified ArrayList
            if (pred != real){ misclassified.add(ex); }
        }
        
        return misclassified;
    }
    
}
