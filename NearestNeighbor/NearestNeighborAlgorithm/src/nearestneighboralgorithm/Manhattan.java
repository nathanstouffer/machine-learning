/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList;

/**
 * class that implements code to compute the distance
 * between two examples using the Manhattan Metric
 * 
 * @author natha
 */
public class Manhattan implements IDistMetric{
    
    /**
     * empty constructor
     */
    Manhattan(){}
    
    /**
     * method to compute the manhattan distance between two points
     * 
     * @param ex1
     * @param ex2
     * @return 
     */
    public double dist(Example ex1, Example ex2){
        // initialize array lists with attributes
        ArrayList<Double> attr1 = ex1.getAttributes();
        ArrayList<Double> attr2 = ex2.getAttributes();
        
        double dist = 0;
        // iterate through attribute arrays
        for (int i = 0; i < attr2.size(); i++){
            // compute the difference between corresponding values
            double diff = (double)(attr2.get(i) - attr1.get(i));
            // add dist to a running total
            dist += diff;
        }
        
        // return the distance squared
        return dist;
    }
    
}
