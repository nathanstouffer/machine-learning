/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList; 
import java.util.Random;
/**
 * Class to reduce the size of a set for use with
 * a Nearest Neighbor algorithm.
 * 
 * CMeans splits a given data set into a specified number of
 * clusters. It then finds the centers, or average values of
 * these clusters. Once centers have been found, all points are
 * reclustered using the new centers. This process is repeated
 * until centers no longer update.
 * @author erick
 */
public class CMeans implements IDataReducer{

    private final Random RD = new Random();                         // MADE these variables private - Stouffer
    private ArrayList<Cluster> clusterList;
    private ArrayList<Example> oldcenters;
    private int c;
    private IDistMetric metric;
    
    /**
     * Constructor to initialize global variables
     * @param k
     * @param metric 
     */
    CMeans(int k, IDistMetric metric){
        this.c = k;
        this.metric = metric;
        this.clusterList = new ArrayList<Cluster>();                // INITIALIZED these with types - Stouff
        this.oldcenters = new ArrayList<Example>();
    }
   
    /**
     * Method to reduce the size of a Set for use with K nearest neighbors
     * @param clone
     * @return 
     */
    @Override
    public Set reduce(Set original){
        // clone the data set
        Set clone = original.clone();                               // CLONED THE DATASET
        // clear ArrayLists
        oldcenters.clear();                                         // CHANGED TO CLEARING ARRAYLIST
        clusterList.clear();
        
        int maxIter = 1000;
        int i = 0;
        initializeClusters(clone);
        populateClusters(clone);
        boolean different = true;
        while (i <= maxIter && different){               //main loop for updating clusters
            for (Cluster cluster : clusterList){         //creates an ArrayList to compare changes made each iteration
                oldcenters.add(cluster.getRep());
            }
            this.updateCenters();                       // ADDED THIS LINE
            //updateCenters(clone);                     // COMMENTED THIS OUT
            populateClusters(clone);
            i++;
            for (int j = 0; j < clusterList.size(); j++){        //loops through all clusters amd checks if they have changed
                if (clusterList.get(j).getRep().getAttributes().equals(oldcenters.get(j).getAttributes())){
                    different = false;
                }
            }
        }
        
        // declare a new set for the reduced data
        // this set will be populated with the centers
        Set reduced = new Set(clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames());
        // iterate through clusters
        for (int j = 0; j < this.c; j++){
            // get jth cluster
            Cluster cluster = this.clusterList.get(j);
            // compute the assigned value of the center
            double center_val = cluster.computeRepValue();
            // get the attributes of the center
            ArrayList<Double> attr = cluster.getRep().getAttributes();
            // instantiate new value
            Example val = new Example(center_val, attr);
            // add val to reduced set
            reduced.addExample(val);
        }
        return reduced;
        
        /*              // COMMENTED THIS OUT. REPLACED WITH ABOVE CODE
        Set reduced = new Set(clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames());  //adds cluster centers to a set to return
        for (Cluster cluster : clusterList){
            reduced.addExample(cluster.getRep());
        }
        return(reduced);
        */
    }
   
    /**
     * Method for randomly initializing random points throughout the data set as new cluster centers
     * @param original 
     */
    private void initializeClusters(Set original){
        int[] used = new int[this.c];                               // MAKE SURE THE RANDOM INDEX HAS NOT BEEN USED
        for (int i = 0; i < c; i++){
            int rnd_index = 0;
            boolean index_used = true;          // assume this is true so first iteration runs                                         
            while (index_used){
                index_used = false;             // assume the index is not used until proven wrong
                // generate random index
                rnd_index = RD.nextInt(original.getNumExamples());
                
                // test if rnd_index is in used
                for (int j = 0; j < i; j++){
                    // set index_used to true if found in unsed_indicies
                    if (rnd_index == used[j]){ index_used = true; }
                }
            }
            
            Example center = original.getExample(rnd_index);
            // Example center = original.getExample(RD.nextInt(original.getNumExamples()));
            center = new Example(center.getValue(), center.getAttributes());                                   // INSTANTIATE THE NEW CENTER WITH ORIGINAL VALUE
            Cluster cluster = new Cluster(center, original.getNumClasses(), this.metric);
            clusterList.add(cluster);
        }
    }
   
    /**
     * Method for populating clusters with examples from the original Set
     * @param original 
     */
    public void populateClusters(Set original){
        for (int i = 0; i < this.c; i++){
        //for (Cluster cluster : clusterList){             //resets clusters so they can be repopulated if already filled     // CHANGED THIS FROM A FOREACH LOOP
            clusterList.get(i).clearCluster();
        }
        for (int i = 0; i < original.getNumExamples(); i++){
        //for(Example ex : original){                                                                             // CHANGED THIS FROM A FOREACH LOOP
            Example ex = original.getExample(i);
            double min = Double.MAX_VALUE;
            Cluster closestCluster = clusterList.get(0);
            for (int j = 0; j < this.clusterList.size(); j++){
            //for (Cluster cluster : clusterList){                 //finds closest cluster to Example ex          // CHANGED THIS FROM A FOREACH LOOP
                double dist = metric.dist(ex, clusterList.get(j).getRep());
                if (dist < min){
                    min = dist;
                    closestCluster = clusterList.get(j);
                }
            }
            closestCluster.addExample(ex);
        }
    }
   
    /**
     * Method for updating cluster centers by taking average value of points in each cluster
     * @param original 
     */
    public void updateCenters(){ // Set original){                        // DELETED Set original PARAMETER
        for (int a = 0; a < this.c; a++){
        //for (Cluster cluster : clusterList){                                                    // CHANGED THIS FROM A FOREACH LOOP
            Cluster cluster = clusterList.get(a);                                                   // ADDED THIS LINE
            Example center = cluster.getRep();
            ArrayList<Double> newAttributes = new ArrayList<Double>(center.getAttributes().size());    //makes an ArrayList to calculate averages
            for (int i = 0; i < center.getAttributes().size(); i++) { //Initialize Arraylist with 0s
                newAttributes.add(0.0);
            }
            for (int i = 0; i < center.getAttributes().size(); i++){
                double currentsum;
                for (int j = 0; j < cluster.getClusterSize(); j++){
                //for (Example ex : original){                                                     //gets the average of each attribute in the cluster    // SEE LINE ABOVE FOR EDIT
                    ArrayList<Double> exAttr = cluster.getExample(j).getAttributes();
                    currentsum = newAttributes.get(i) + exAttr.get(i);
                    newAttributes.set(i, currentsum);
                }
                double average = newAttributes.get(i)/cluster.getClusterSize();
                newAttributes.set(i, average);
            }
            Example newCenter = new Example(-1, newAttributes);              //puts the averages into an example to save as the new center // INSTANTIATE CENTERS WITH INVALID VALUE
            clusterList.get(a).setRep(newCenter);
            //System.out.println("NUM EXAMPLES IN CLUSTER: " + Integer.toString(clusterList.get(a).getClusterSize()));
        }
    }
}

