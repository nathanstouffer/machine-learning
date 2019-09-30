/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList;

/**
 * class that implements code to compute the distance
 * between two examples using the square of the 
 * Euclidean Metric
 * 
 * @author natha
 */
public class EuclideanSquared implements IDistMetric {
    
    /**
     * empty constructor
     */
    public EuclideanSquared(){}
    
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
        ArrayList<Double> attr1 = ex1.getAttributes();
        ArrayList<Double> attr2 = ex2.getAttributes();
        
        double sqrd_dist = 0;
        // iterate through attribute arrays
        for (int i = 0; i < attr2.size(); i++){
            // compute the difference between corresponding values
            double diff = (double)(attr2.get(i) - attr1.get(i));
            // square the difference
            double sqrd_diff = (diff * diff);
            // add sqrd_diff to a running total
            sqrd_dist += sqrd_diff;
        }
        
        // return the distance squared
        return sqrd_dist;
    }
    
}
