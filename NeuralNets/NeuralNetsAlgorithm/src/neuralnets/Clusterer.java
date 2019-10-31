/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.*;
import reducedata.*;
import measuredistance.EuclideanSquared;
import java.util.ArrayList;

/**
 *
 * @author erick
 */

public class Clusterer {
    
    // distance euclidean used in computation
    private final EuclideanSquared euclidean;
    // original data set
    private Set orig;
    // original subsets
    private Set[] subsets;
    // number of neighbors to consider when computing variance
    private int k;
    // cluster array of the form { Edited/Condensed, CMeans, CMedoids }
    private Set[] reps;
    // variance array of the form { Edited/Condensed, CMeans, CMedoids }
    // where each index contains an array of variances corresponding
    // to a representative
    private double[][] vars;
    
    /**
     * constructor to instantiate a Cluster object with a EuclideanSquared
     * distance metric
     * 
     * @param euc Euclidean distance metric to use
     */
    public Clusterer(EuclideanSquared euc) {
        // initialize global variables
        this.euclidean = euc;
        // insantiate to correct size
        this.reps = new Set[3];
        this.vars = new double[3][];
    }
    
    /**
     * method to cluster the data and compute the variance for
     * each cluster
     * @param subsets subsets that compose the data to be clustered
     * @param k number of neighbors to consider when computing variance
     */
    public void cluster(Set[] subsets, int k) {
        // number of neighbors to consider when computing variance
        this.k = k;
        // subsets that compose dataset
        this.subsets = subsets;
        // build entire data set from subsets
        // pass in -1 so entire set is constructed
        this.orig = new Set(this.subsets, -1);
        
        // determine if data set is classification or regression
        boolean regression = false;     // assume data set is not regression
        if (orig.getNumClasses() == -1) { regression = true; }
        
        // compute reps
        this.computeReps(regression);
        
        // compute variances for each set of representatives
        if (regression) { this.vars[0] = null; }
        else { this.vars[0] = this.computeVariances(this.reps[0]); }
        this.vars[1] = this.computeVariances(this.reps[1]);
        this.vars[2] = this.computeVariances(this.reps[2]);
    }
    
    /**
     * method to compute the clusters for each algorithm
     * @param regression indicator of whether dataset is regression
     */
    private void computeReps(boolean regression) { 
        // variable to determine how many reps CMeans and CMedoids should use
        // initially set to 1/4 the size of the data set
        int num_clust = (int) (0.25 * this.orig.getNumExamples());
        
        // data reducer used
        IDataReducer reducer;
        if (regression) { this.reps[0] = null; }
        // otherwise, dataset is classification
        else { 
            // edited requires a validation_set set, this is partitioned here
            Set validation_set = this.subsets[0];
            Set edited_orig = new Set(this.subsets, 0);
            // instantiate edited
            // 1 is used as the number of neighbors to consider
            // while reducing the data set
            reducer = new Edited(this.euclidean, 1, validation_set);
            // reduce edited_orig and store in reps[0]
            this.reps[0] = reducer.reduce(edited_orig);
            
            // instantiate condensed
            reducer = new Condensed(this.euclidean);
            // reduce orig
            Set reduced = reducer.reduce(this.orig);
            // store in clusers[0] if condensed's size is smaller
            if (reduced.getNumExamples() < this.reps[0].getNumExamples()) { 
                this.reps[0] = reduced;
            }
            
            // set num_clust to size of reduced data set
            num_clust = this.reps[0].getNumExamples();
        }
        
        // instantiate CMeans
        reducer = new CMeans(this.euclidean, num_clust);
        // reduce orig and store in reps[1]
        this.reps[1] = reducer.reduce(this.orig);
        
        // instantiate CMedoids
        reducer = new CMedoids(this.euclidean, num_clust);
        // reduce orig and store in reps[2]
        this.reps[2] = reducer.reduce(this.orig);
    }

    /**
     * method to compute and return the variances for each representative
     * in a set of representatives
     * 
     * @param reps set of representatives
     */
    private double[] computeVariances(Set reps) { 
        // instantiate vars array with correct size
        double[] vars = new double[reps.getNumExamples()];

        // iterate through representatives
        for (int i = 0; i < reps.getNumExamples(); i++) { 
            // get current representative
            Example curr_rep = reps.getExample(i);
            // find k-nearest neighbor of curr_rep
            ArrayList<Example> neighbors = this.computeKNeighbors(curr_rep);

            // compute variance of curr_rep
            double variance = 0.0;
            // add squared distance from curr_rep to variance
            for (int j = 0; j < this.k; j++) { 
                variance += this.euclidean.dist(curr_rep, neighbors.get(j)); 
            }
            variance /= this.k;
            variance = Math.sqrt(variance);

            // store variance in array
            vars[i] = variance;
        }
        
        // return vars
        return vars;
    }
    
    /**
     * method to find the k nearest neighbors of a representative
     * @param rep
     * @return 
     */
    private ArrayList<Example> computeKNeighbors(Example rep) {
        // k nearest neighbors
        ArrayList<Example> neighbors = new ArrayList<Example>(this.k);
        // distance from rep for corresponding neighbor
        ArrayList<Double> distances = new ArrayList<Double>(this.k);
        
        // initialize all distances to Double.MAX_VALUE
        for (int i = 0; i < distances.size(); i++) { distances.set(i, Double.MAX_VALUE); }
        
        // iterate through entire dataset
        for (int i = 0; i < this.orig.getNumExamples(); i++) {
            Example ex = this.orig.getExample(i);
            
            // compute distance between rep and ex
            double dist = this.euclidean.dist(rep, ex);
            
            // variable to tell whether the example has been added to the neighbors
            boolean added = false;
            // iterate through neighbors
            for (int j = 0; j < distances.size() && !added; j++) {
                // test if dist is less than any current neighbor
                if (dist < distances.get(j)) {
                    // update neighbors and distances ArrayLists
                    neighbors.add(i, ex);
                    distances.add(i, dist);
                    // remove last element in each ArrayList
                    neighbors.remove(this.k);
                    distances.remove(this.k);
                    // set added to true
                    added = true;
                }
            }
        }
        
        // return neighbors
        return neighbors;
    }
    
    public Set[] getReps() { return this.reps; }
    public double[][] getVars() { return this.vars; }

}