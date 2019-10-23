/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reducedata;

import java.util.ArrayList;
import java.util.Random;
import datastorage.Example;
import measuredistance.IDistMetric;
import datastorage.Set;

/**
 * Class implement K medoids algorithm and outputs Set of medoids for use by KNN algorithm
 * @author Kevin
 */
public class CMedoids implements IDataReducer {

    //number of clusters to create, gets from ENN
    private int c;
    //initilize computeDistortion variable
    private double distortion;
    // metric that the reduciing algorithm should use
    private IDistMetric metric;
    //Euclidean squared object to get distances between points
    private Cluster[] clusters;
    
    /**
     * constructor takes number of medoids c and distance metric to use
     * @param c
     * @param met 
     */
    public CMedoids(int c, IDistMetric met) {
        this.c = c;
        this.metric = met;
        this.distortion = 0.0;
        this.clusters = new Cluster[c];
    }
    /**
     * Implements reduce method from IDataReducer and outputs a reduced data set
     * @param orig
     * @return 
     */
    @Override
    public Set reduce(Set orig) {
        //clones original data set in order to delete from set
        Set clone = orig.clone();
        
        // initialize the medoids at random values
        this.initializeMedoids(clone);
                
        // boolean to determine if a swap occured
        boolean change = true;
        // loop to run until no change happens or 1000 loops
        for (int g = 0; g < 1000 && change; g++) {            
            // clear the current clusters
            for (int i = 0; i < this.c; i++){ this.clusters[i].clearCluster(); }

            // cluster the data using the current medoids
            this.cluster(clone, this.getMedoids());
            
            // find the best fitting medoid in each cluster
            change = this.swapMedoids();            
        }
        
        // declare a new set for the reduced data
        // this set will be populated with the medoids
        Set reduced = this.computeReducedSet(clone);
        return reduced;
    }
    
    /**
     * method to swap the each medoid with each example in its cluster
     * returns false if medoids do not change
     * @return 
     */
    private boolean swapMedoids(){
        // assume that the current medoids are on the final iteration
        boolean change = false;
        // declare distortion array
        double[] distortions = new double[this.c];
        // populate distortions array with distortions
        for (int i = 0; i < this.c; i++){ distortions[i] = this.clusters[i].computeDistortion(); }
        
        // test if any medoid should be swapped
        for (int i = 0; i < this.c; i++){
            // current cluster and medoid
            Cluster cluster = this.clusters[i];
            // iterate through each point in the current cluster
            for (int j = 0; j < cluster.getClusterSize(); j++){
                // current medoid
                Example medoid = cluster.getRep();
                // jth example in cluster
                Example ex = cluster.getExample(j);

                // swap ex with the current medoid
                cluster.replaceExample(j, medoid);
                cluster.setRep(ex);

                // compute new computeDistortion for this cluster
                double new_distortion = cluster.computeDistortion();

                // compare to original computeDistortion
                if (distortions[i] <= new_distortion){
                    // swap back to original medoid
                    cluster.replaceExample(j, ex);
                    cluster.setRep(medoid);
                }
                else {
                    // a medoid was correctly swapped, so change must be true
                    change = true;
                    // update the distortions array
                    distortions[i] = new_distortion;
                }
            }
            //System.out.println("NUM EXAMPLES IN CLUSTER: " + Integer.toString(cluster.getClusterSize()));
        }
        
        return change;
    }
    
    /**
     * method to create a set from medoid of each cluster
     * @param dataset
     * @return 
     */
    private Set computeReducedSet(Set dataset){
        Set reduced = new Set(dataset.getNumAttributes(), dataset.getNumClasses(), dataset.getClassNames());
        // final medoids
        Example[] medoids = this.getMedoids();
        // iterate through medoids
        for (int i = 0; i < this.c; i++){
            // compute the assigned value of the medoid
            double medoid_val = this.clusters[i].computeRepValue();
            // get the attributes of the medoid
            ArrayList<Double> attr = medoids[i].getAttributes();
            // instantiate new value
            Example val = new Example(medoid_val, attr);
            // add val to reduced set
            reduced.addExample(val);
        }
        // return dataset consisting of only medoids
        return reduced;
    }
    
    /**
     * method to cluster the data around the medoids
     * @param data
     * @param medoids 
     */
    private void cluster(Set data, Example[] medoids){
        // iterate through each example in the data
        for (int i = 0; i < data.getNumExamples(); i++) {
            // example we must assign to a medoid
            Example ex = data.getExample(i);
            // index of closest medoid
            int min_index = 0;

            // assume current distance from ex to closest medoid is Double.MAX_VALUE
            double min_dist = Double.MAX_VALUE;
            // iterate through medoids
            for (int a = 0; a < this.c; a++) {
                // get current medoid
                Example curr_medoid = medoids[a];
                // compute distance between ex and medoid
                double dist = metric.dist(ex, curr_medoid);
                // compare dist to min_dist
                if (dist < min_dist) {
                    // if dist is less than min_dist, make dist the new min_dist
                    min_dist = dist;
                    min_index = a;
                }
            }
            // assign ex to a medoid
            this.clusters[min_index].addExample(ex);
        }
    }
    
    /**
     * method to return an Example array with all the medoids
     * @return 
     */
    private Example[] getMedoids(){
        // declare correctly sized array
        Example[] medoids = new Example[this.c];
        
        // iterate through each cluster
        for (int i = 0; i < c; i++){ 
            // add cluster representative to medoids
            medoids[i] = this.clusters[i].getRep();
        }
        
        return medoids;
    }
    
    /**
     * method to randomly declare the medoids
     * @param data 
     */
    private void initializeMedoids(Set data){
        // variable for the number of classes in clone
        int num_classes = data.getNumClasses();
        
        // add c number of random examples as medoids to clusters array
        Random rand = new Random();
        for (int i = 0; i < this.c; i++) {
            // compute random index
            int rand_index = rand.nextInt(data.getNumExamples());
            // create medoid from value at rand_index in data
            Example medoid = data.getExample(rand_index);
            
            // create new cluster with medoid
            Cluster temp = new Cluster(medoid, num_classes, this.metric);
            // add the new cluster to clusters
            this.clusters[i] = temp;
            
            // delete medoid from data so it will not be chosen again
            data.delExample(medoid);
        }
    }
}
