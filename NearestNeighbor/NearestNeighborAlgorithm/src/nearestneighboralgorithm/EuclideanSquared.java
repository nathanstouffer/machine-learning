/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList;

/**
 *
 * @author natha
 */
public class EuclideanSquared implements IMetric {
    
    EuclideanSquared(){}
    
    /**
     * method to compute the squared distance between two points
     * 
     * this method exists for computation efficiency of finding
     * the minimum distance 
     * 
     * @param ex1
     * @param ex2
     * @return 
     */
    public double dist(Example ex1, Example ex2){
        // initialize array lists with attributes
        ArrayList<Integer> attr1 = ex1.getAttributes();
        ArrayList<Integer> attr2 = ex2.getAttributes();
        
        /**
         * NOTES
         * 1: figure out the double vs int casting
         * 2: consider using logs to get rid of floating point errors
         */
        ArrayList<Double> sqrd_diff = new ArrayList<Double>();
        // iterate through attribute arrays
        for (int i = 0; i < attr1.size(); i++){
            // compute the squared difference between corresponding values
            double diff = (double)(attr1.get(i) - attr2.get(i));
            sqrd_diff.add(diff * diff);
        }
        
        // sum the squared differences
        double sqrd_dist = 0;
        for (Double d: sqrd_diff){ sqrd_dist += d; }
        
        // return the distance squared
        return sqrd_dist;
    }
    
}
