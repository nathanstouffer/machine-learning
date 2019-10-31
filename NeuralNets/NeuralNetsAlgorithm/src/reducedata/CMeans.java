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
 *
 * @author natha
 */
public class CMeans implements IDataReducer {
    
    // number of clusters
    private final int C;
    // distance metric
    private IDistMetric metric;
    // clusters in the data set
    private Cluster[] clusters;
    // maximum difference in distances to be considered converged
    private final double CONVERGENCE_LEVEL = 0.01;
    
    public CMeans(IDistMetric metric, int c){
        this.C = c;
        this.metric = metric;
    }
    
    public Set reduce(Set orig){
        // clone set
        Set clone = orig.clone();
        
        // randomly initialize centroids
        this.initializeClusters(clone);
        
        // variable to tell whether the means have converged
        boolean converged = false;
        // start for loop here
        for (int i = 0; i < 1000 && !converged; i++){
            // clear clusters
            for (int j = 0; j < this.C; j++){ this.clusters[j].clearCluster(); }
            
            // populate clusters
            this.populateClusters(clone);

            // calculate new centroids based on current clusters
            Example[] new_centroids = this.computeNewCentroids(clone.getNumAttributes());

            // assume that the centroids have converged
            converged = true;
            // test convergence and update centroids
            for (int j = 0; j < this.C; j++){
                // test convergence
                double dist = metric.dist(new_centroids[j], clusters[j].getRep());
                if (dist > this.CONVERGENCE_LEVEL){ converged = false; }
                // update centroid
                clusters[j].setRep(new_centroids[j]);
            }
        }
        
        // instantiate and populate new set with just centroids
        Set reduced = this.computeReducedSet(clone);
        return reduced;
    }
    
    private Example[] computeNewCentroids(int num_attr){
        // array for new means
        Example[] new_centroids = new Example[this.C];
        // iterate through each cluster, computing the mean
        for (int i = 0; i < this.C; i++){
            // current cluster
            Cluster cluster = this.clusters[i];
            
            // compute new centroid
            Example centroid = this.computeNewCentroid(cluster, num_attr);
            // add mean into new_centroids
            new_centroids[i] = centroid;
        }
        
        return new_centroids;
    }
    
    /**
     * method to compute a single new centroid of a given cluster
     * @param cluster
     * @param num_attr
     * @return 
     */
    private Example computeNewCentroid(Cluster cluster, int num_attr){
        // ArrayList to hold the mean of each attribte in a cluster
        // this will be the new centroid of the cluster
        ArrayList<Double> mean = new ArrayList<Double>(num_attr);
        // initialize all values in mean to 0.0
        for (int j = 0; j < num_attr; j++){ mean.add(0.0); }
        
        // iterate through each example in the cluster
        // loop takes the sum of each attribute value in preparation for computing the average
        for (int j = 0; j < cluster.getClusterSize(); j++){
            // current example in cluster
            Example ex = cluster.getExample(j);
            // add the kth attribute value to the kth attribute value in mean
            for (int k = 0; k < num_attr; k++){
                double prev = mean.get(k);
                double to_add = ex.getAttributes().get(k);
                mean.add(k, prev + to_add);
            }
        }
        // divide each sum by the number of examples in the cluster
        for (int j = 0; j < num_attr; j++){
            double orig = mean.get(j);
            mean.set(j, orig / cluster.getClusterSize());
        }
        
        // instantiate a centroid with mean as the attributes
        Example centroid = new Example(-1, mean);
        return centroid;
    }
    
    /**
     * method to create a set from medoid of each cluster
     * @param dataset
     * @return 
     */
    private Set computeReducedSet(Set dataset){
        Set reduced = new Set(dataset.getNumAttributes(), dataset.getNumClasses(), dataset.getClassNames());
        // iterate through medoids
        for (int i = 0; i < this.C; i++){
            // compute the assigned value of the centroid
            double centroid_val = this.clusters[i].computeRepValue();
            // get the attributes of the centroid
            ArrayList<Double> attr = this.clusters[i].getRep().getAttributes();
            // instantiate new value
            Example val = new Example(centroid_val, attr);
            // add val to reduced set
            reduced.addExample(val);
        }
        // return dataset consisting of only centroids
        return reduced;
    }
    
    /**
     * method to populate clusters
     * @param dataset 
     */
    private void populateClusters(Set dataset){
        // iterate through each example in the dataset
        for (int i = 0; i < dataset.getNumExamples(); i++){
            // current example
            Example ex = dataset.getExample(i);
            // defining variables for closest mean
            double min_dist = Double.MAX_VALUE;
            int closest_index = 0;
            // iterate through the centroids, finding the closest mean
            for (int j = 1; j < this.C; j++){
                // current centroid
                Example centroid = this.clusters[j].getRep();
                // compute new distance
                double new_dist = this.metric.dist(ex, centroid);
                // compare distances
                if (new_dist < min_dist){
                    closest_index = j;
                    min_dist = new_dist;
                }
            }
            
            // add example to the cluster of closest mean
            Cluster closest = this.clusters[closest_index];
            closest.addExample(ex);
        }
    }
    
    /**
     * method to initialize all clusters with random means
     * @param dataset 
     */
    private void initializeClusters(Set dataset){
        // array storing the indices used for initial means
        int[] used = new int[this.C];
        // loop to initialize the initial means
        for (int i = 0; i < this.C; i++){
            // generate a new cluster index
            int new_cluster_index = this.getRandIndex(used, i, dataset.getNumExamples());
            used[i] = new_cluster_index;
            // initialize mean as the random point
            Example mean = dataset.getExample(new_cluster_index);
            // centroids have no value until it is assigned, thus -1
            mean = new Example(-1, mean.getAttributes());
            // create new cluster centered on mean
            Cluster cluster = new Cluster(mean, dataset.getNumClasses(), this.metric);
            // insert cluster into clusters array
            this.clusters[i] = cluster;
        }
    }
    
    /**
     * method to return an unused random index
     * 
     * curr_num_clusters describes the number 
     * @param used
     * @param curr_num_clusters
     * @param num_examples
     * @return 
     */
    public int getRandIndex(int[] used, int curr_num_clusters, int num_examples){
        // random index
        int rnd_index = 0;
        
        // random number generator
        Random rand = new Random();
        
        boolean index_used = true;                                         
        // ensure the random index has not been used
        while (index_used){
            // assume the index is not used until proven wrong
            index_used = false;             
            // generate random index
            rnd_index = rand.nextInt(num_examples);

            // test if rnd_index is in used
            for (int j = 0; j < curr_num_clusters; j++){
                // set index_used to true if found in unsed_indicies
                if (rnd_index == used[j]){ index_used = true; }
            }
        }
        
        // return the generated index
        return rnd_index;
    }
    
}
