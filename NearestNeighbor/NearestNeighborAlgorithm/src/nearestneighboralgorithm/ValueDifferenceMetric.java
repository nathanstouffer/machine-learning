/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 * Class to compute the distance between two categorical
 * attributes
 * 
 * This class implements the Value Difference Metric
 * to compute the distance
 *
 * @author natha
 */
public class ValueDifferenceMetric {
    
    /**
     * method that implements the Value Difference Metric to compute
     * the distance between two categorical attributes
     * @param option1
     * @param option2
     * @param mtr
     * @return 
     */
    public static double dist(int option1, int option2, SimilarityMatrix mtr){
        // if the options are equal, there is no difference
        if (option1 == option2){ return 0.0; }
        // otherwise, compute the difference
        else{
            // stores the distance between two options
            double dist = 0.0;
            // iterate through each class
            for (int classification = 0; classification < mtr.getNumBins(); classification ++){
                // compute difference in each class
                double diff = mtr.getProb(option2, classification) - mtr.getProb(option1, classification);
                
                // take the absolute value of diff
                diff = Math.abs(diff);
                
                // optional: square diff
                // diff = Math.pow(diff, 2);
                
                // add to running total
                dist += diff;
            }
            // regularize the distance by dividing by the number of classes
            dist /= mtr.getNumBins();   // THIS MAY NOT BE A GOOD IDEA
            // return the distance
            return dist;
        }
    }
    
}
