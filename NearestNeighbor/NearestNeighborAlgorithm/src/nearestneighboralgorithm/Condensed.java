/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.lang.Double;
import java.util.Random;
import java.util.Iterator;


/**
 * Class to reduce the size of a set for use with a 
 * Nearest Neighbor algorithm.
 * 
 * The Condensed Nearest Neighbor algorithm looks at the given
 * data set and condenses points of the same class that neighbor
 * each other into a single point, decreasing total number of points
 * in the set. This process is repeated until the set does not change.
 * @author erick
 */
public class Condensed {
    
    // variable to determine how many neighbors the learning
    // algorithm will view
    private int k;
    // metric that the reduciing algorithm will use
    private IDistMetric metric;
    Random rd = new Random();

 
    /**
     * Constructor to initialize a new instance of Condensed

     * @param metric 
     */
    Condensed(IDistMetric metric){
        this.metric = metric;
    }
    
    /**
     * Method that takes in a set and returns the reduced version after running the condensing algorithm on it
     * @param original
     * @return 
     */
    public Set reduce(Set original){
        
        Set reduced = new Set(original.getNumAttributes(), original.getNumClasses(), original.getClassNames());
        reduced.addExample(original.getExample(rd.nextInt(original.getNumExamples())));
        Set oldSet = original.clone();
        Set copy = original.clone();
        int maxIter = 1000;
        int i = 0;
        while (!reduced.getExamples().equals(oldSet.getExamples()) && i <= maxIter){        //while the set changes from one iteration to the next
            oldSet = reduced.clone();
            for (int j = 0; j < copy.getNumExamples(); j++) {
                Example ex = copy.getExample(j);
                double min = Double.MAX_VALUE;                                          //saving minimums to find the closest example in the set, for each example
                Example minEx = copy.getExample(0);
                for (int k = 0; k < reduced.getNumExamples(); k++) {
                    Example ex2 = reduced.getExample(k);
                    if (!ex.getAttributes().equals(ex2.getAttributes())){               //checking for identical example(self)
                        if (metric.dist(ex, ex2) < min){                                //checking if distance to ex2 is smaller than minimum so far
                            min = metric.dist(ex, ex2);
                            minEx = ex2;
                        }
                    }
                }
                if (minEx.getValue() != ex.getValue()){                     //adds the closest example to the reduced set if it has different class than 
                    reduced.addExample(ex);                              //example we are comparing to
                    copy.delExample(j);
                    j--;
                }
            }
            i++;
        }
        return reduced;
    }
    
}
