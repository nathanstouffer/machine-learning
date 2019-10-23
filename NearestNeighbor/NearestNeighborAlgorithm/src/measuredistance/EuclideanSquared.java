/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package measuredistance;

import datastorage.Example;
import java.util.ArrayList;
import datastorage.SimilarityMatrix;

/**
 * Class that implements code to compute the distance
 * between two examples using the square of the 
 * Euclidean Metric
 * 
 * @author natha
 */
public class EuclideanSquared implements IDistMetric {
    
    private SimilarityMatrix[] sim_matr;
    
    /**
     * constructor to initialize global variable
     */
    public EuclideanSquared(SimilarityMatrix[] sim_matr){ this.sim_matr = sim_matr; }
    
    /**
     * method to compute the squared Euclidean distance between two examples
     * 
     * @param ex1
     * @param ex2
     * @return 
     */
    public double dist(Example ex1, Example ex2){
        // initialize array lists with attributes
        ArrayList<Double> attr1 = ex1.getAttributes();
        ArrayList<Double> attr2 = ex2.getAttributes();
        
        double sqrd_dist = 0.0;
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
                
                // call categorical dist function (computes difference between corresponding values)
                diff = ValueDifferenceMetric.dist(option1, option2, this.sim_matr[cat_index]);
            }
            else{
                // compute the difference between corresponding values
                diff = (double)(attr2.get(i) - attr1.get(i));
            }
            
            // square diff
            double sqrd_diff = Math.pow(diff, 2);
            //System.out.println(sqrd_diff);
            // add dist to a running total
            sqrd_dist += sqrd_diff;
        }
        
        //System.out.println("FINAL DIST: " + sqrd_dist);
        // return the squared Euclidean distance
        return sqrd_dist;
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
