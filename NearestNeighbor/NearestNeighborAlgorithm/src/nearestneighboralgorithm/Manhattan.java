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
    
    private SimilarityMatrix[] sim_matr;
    
    /**
     * constructor to initialize global variable
     */
    Manhattan(SimilarityMatrix[] sim_matr){ this.sim_matr = sim_matr; }
    
    /**
     * method to compute the Manhattan distance between two points
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
            // variable to store difference in attributes
            double diff = 0.0;
            // test if attribute is categorical
            int cat_index = this.categoricalIndex(i);
            if (cat_index > -1){
                // convert inputs to integer indices for a SimilarityMatrix
                int option1 = (int)Math.round(attr1.get(i));
                int option2 = (int)Math.round(attr2.get(i));
                
                // call categorical dist function
                diff = Categorical.dist(option1, option2, this.sim_matr[cat_index]);
            }
            else{
                // compute the difference between corresponding values
                diff = (double)(attr2.get(i) - attr1.get(i));
            }
            
            // add dist to a running total
            dist += diff;
        }
        
        // return the manhattan distance
        return dist;
    }
    
    /**
     * method to determine whether a given attribute contains
     * categorical data
     * @param index
     * @return 
     */
    public int categoricalIndex(int index){
        for (int i = 0; i < this.sim_matr.length; i++){
            if (index == this.sim_matr[i].getAttrIndex()){ return i; }
        }
        return -1;
    }
    
}
