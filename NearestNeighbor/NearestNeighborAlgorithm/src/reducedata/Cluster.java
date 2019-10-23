/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reducedata;

import datastorage.Example;
import java.util.ArrayList;
import knearestneighbor.measuredistance.IDistMetric;

/**
 * Cluster class has a rep example and set of points in it examples
 used for simplifying clustering algorithms
 * @author Kevin
 */
public class Cluster {
    //representative example
    private Example rep;
    //distance metric used to calculate computeDistortion for C Medoids
    private IDistMetric metric;
    //cluster of points under rep point
    private ArrayList<Example> examples;
    // variable to tell how many classes the data set has
    // -1 implies regression
    private int num_classes;
    
    /**
     * constructor takes in a rep example, and set attributes to make a new
     * set for that example
     * @param rep
     * @param regression
     * @param dist_met
     */
    public Cluster(Example rep, int num_classes, IDistMetric dist_met) {
        this.metric = dist_met;
        this.rep = rep;
        
        // instantiate the examples ArrayList
        this.examples = new ArrayList<Example>();
        
        // set the average of global variable num_classes
        this.num_classes = num_classes;
    }
        
    /**
     * calculates computeDistortion for C medoids
     * @return 
     */
    public double computeDistortion(){
        // set initial value to 0
        double distortion = 0;
        // sum the distortion of each example with respect to to
        for (int i = 0; i < this.examples.size(); i++){
            distortion += this.metric.dist(this.rep, this.examples.get(i));
        }
        // return final value
        return distortion;
    }
    
    /**
     * method to return the average that the representative of the cluster takes on. 
     * @return 
     */
    public double computeRepValue(){
        // test if set is null
        // end computation if true
        if (this.examples.size() == 0){
            //System.err.println("No examples in cluster.\nRep value: "
                    //+ Double.toString(this.rep.getValue()));
            return this.rep.getValue();
        }
        
        double assigned_value = 0.0;
        // if data set is regression, compute the average of all values in the cluster
        if (this.num_classes == -1){
            // compute regression value for medoid
            assigned_value = this.computeRegressionValue();
        }
        // otherwise, have all examples in the cluster "vote" for a class
        else{ this.computeClassificationValue(); }
        
        // return the assigned average of the cluster
        return assigned_value;
    }
    
    /**
     * private method to compute the assigned class for the representative
     * @return 
     */
    private double computeClassificationValue(){
        // declare an array to store the count of each class in the cluster
        int[] class_count = new int[this.num_classes];

        // iterate through the examples, incrementing the class_count array
        for (int i = 0; i < this.examples.size(); i++){
            // get classification
            int classification = (int)this.examples.get(i).getValue();
            // increment class_count at classification
            class_count[classification]++;
        }

        // discern between medoids and means
        if (this.rep.getValue() != -1){
            // acount for rep in "vote"
            // get classification
            int classification = (int)this.rep.getValue();
            // increment class_count at classification
            class_count[classification]++;
        }

        // find max of class_count
        // assume the max is at 0
        int max_index = 0;
        for (int i = 1; i < class_count.length; i++){
            // swap if ith average is greater than max
            if (class_count[i] > class_count[max_index]){ max_index = i; }
        }

        // return the max_index
        return max_index;
    }
    
    /**
     * private method to compute the assigned regression value for the representative
     * @return 
     */
    private double computeRegressionValue(){
        // average
        double average = 0.0;
        // sum the values of each example in the cluster
        for (int i = 0; i < this.examples.size(); i++){ average += this.examples.get(i).getValue(); }
        
        // discern between medoids and means
        if (this.rep.getValue() != -1){
            // account for value of medoid in average
            average += this.rep.getValue();
            // divide by the count of examples + 1
            average /= (this.examples.size() + 1);
        }
        // otherwise just compute average
        else{
            // divide by the count of examples
            average /= this.examples.size();
        }
        return average;
    }
    
    /**
     * method to replace the example at index with the argument ex
     * @param index
     * @param ex 
     */
    public void replaceExample(int index, Example ex){
        // remove old example
        this.examples.remove(index);
        // insert new example at index
        this.examples.add(index, ex);
    }
    
    //getters and setters for examples and rep
    public void clearCluster(){ this.examples.clear(); }
    public void addExample(Example example){ this.examples.add(example); }
    public void addExample(int index, Example example){ this.examples.add(index, example); }
    public void delExample(Example example){ this.examples.remove(example); }
    public void setRep(Example example){ this.rep = example; }
    public int getClusterSize(){ return this.examples.size(); }
    public Example getRep(){ return this.rep; }
    public Example getExample(int index){ return this.examples.get(index); }
    public ArrayList<Example> getExamples(){ return this.examples; }
    
}
