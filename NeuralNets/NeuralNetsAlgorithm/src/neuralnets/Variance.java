/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.Example;
import datastorage.Set;
import java.util.ArrayList;
import measuredistance.IDistMetric;
import java.util.Random;
/**
 *
 * @author erick
 */

public class Variance {
    
    private IDistMetric metric;
    private int c;
    Random rd;
    
    public Variance(IDistMetric metric, int c){
        
        //distance metric to use
        this.metric = metric;
        //number of points to find variance with
        this.c = c;
        //new random instance
        this.rd = new Random();
        
        
    }

/**
 * Method for computing variance between a set and a reduced set for a
 * specified number of examples.
 * @param data
 * @param reduced
 * @param numAttr
 * @return 
 */
    public double computeVariance(Set data, Set reduced, int numAttr){
        Set full = data.clone();
        ArrayList<Double> mean = new ArrayList<Double>(numAttr);
        ArrayList<Example> varianceList = new ArrayList<Example>();
        Example centerVariance = reduced.getExample(rd.nextInt(reduced.getNumExamples()));
        for (int i = 0; i < numAttr; i++){
            //initializes mean array to 0s
            mean.add(0.0);
        }
        for (int i = 0; i < c; i++){
            //Finding c number of closest examples from original set
            Example min = data.getExample(0);
            for (Example ex : full.getExamples()){
                //Looking through examples from full set to find the next closest to center
                if (metric.dist(centerVariance, ex) < metric.dist(centerVariance, min)){
                    min = ex;
                }
            }
            full.rmExample(min);
            varianceList.add(min);
            //moving closest example into variance cluster
            for (int j = 0; j < numAttr; j++){
                //Finding the mean value of variance cluster (needed for variance calculation)
                double newValue = mean.get(j) + min.getAttributes().get(j);
                mean.set(j, newValue);
            }
        }
        double variance = 0.0;
        for (Example ex : varianceList){
            //Calculating variance for all points in varianceList
            ArrayList<Double> attributes = ex.getAttributes();
            for (int i = 0; i < attributes.size(); i++){
                double difference = attributes.get(i) - mean.get(i);
                variance += difference * difference;
            }
        }
        variance /= c;
        return variance;
    }

}