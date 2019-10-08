/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.lang.Double;
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
 
    
    Condensed(int k, IDistMetric metric){
        this.k = k;
        this.metric = metric;
        this.learner = new KNNClassifier();
        this.learner.setK(k);
        this.learner.setDistMetric(metric);
    }
    
    public Set reduce(Set original){
        
        Set reduced = new Set(original.getNumAttributes(), original.getNumClasses(), original.getClassNames());
        Set oldSet = reduced.clone();
        int maxIter = 1000;
        int i = 0;
        while (!reduced.getExamples().equals(oldSet.getExamples()) && i <= maxIter){
            for(Example ex: reduced){
                double min = Double.MAX_VALUE;
                Example minEx = original.getExample(0);
                for (Example ex2 : reduced){
                    if (!ex.getAttributes().equals(ex2.getAttributes())){               //checking for identical example(self)
                        if (metric.dist(ex, ex2) < min){                                //checking if distance to ex2 is smaller than minimum so far
                            min = metric.dist(ex, ex2);
                            minEx = ex2;
                        }
                    }
                }
                if (minEx.getClass() != ex.getClass()){
                    reduced.addExample(minEx);
                }
            }
            i++;
        }
        return reduced;
    }
    
}
