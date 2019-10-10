/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

/**
 * Class implement K medoids algorithm and outputs Set of medoids for use by KNN algorithm
 * @author Kevin
 */
public class CMedoids implements IDataReducer {

    //number of clusters to create, gets from ENN
    private int c;
    //initilize distortion variable
    private double distortion = 0;
    // metric that the reduciing algorithm should use
    private IDistMetric metric;
    //Euclidean squared object to get distances between points
    private ArrayList<Cluster> clusters = new ArrayList<Cluster>();
    /**
     * constructor takes number of medoids c and distance metric to use
     * @param c
     * @param met 
     */
    CMedoids(int c, IDistMetric met) {
        this.c = c;
        this.metric = met;
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

        //add c number of random examples as medoids to clusters array
        for (int i = 0; i < c; i++) {
            int dist = new Random().nextInt(clone.getNumExamples());
            clusters.add(new Cluster(clone.getExample(dist), clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames(), metric));
            clone.delExample(clone.getExample(dist));
        }
        //boolean to determine if a swap occured
        boolean change = true;
        //loop to run until no change happens or 1000 loops
        for (int g = 0; g < 1000 && change; g++) {
            change = false;
            //adding each point to the cluster of its nearest medoid
            for (int i = 0; i < clone.getNumExamples(); i++) {
                //setting min value
                double min = Double.MAX_VALUE;
                //index of closest medoid
                int minmedoidindex = 0;
                Example example = clone.getExample(i);
                //getting distance between each example and each medoid
                for (int a = 0; a < clusters.size(); a++) {
                    Example medoid = clusters.get(a).getRepresentative();
                    double dist = metric.dist(example, medoid);
                    //if distance is less change min distance and closest medoid index
                    if (dist < min) {
                        min = dist;
                        minmedoidindex = a;
                    }
                }
                clusters.get(minmedoidindex).AddExample(example);
            }
            //calculate total distortion
            for (Cluster c : clusters) {
                distortion += c.distortion();
            }
            //swapping each medoid with all points in cluster
            for (int a = 0; a < clusters.size(); a++) {
                //iterating through all example by cluster
                for (Cluster c : clusters) {
                    for (int e = 0; e < c.getCluster().getNumExamples(); e++) { //Example e : c.getCluster()) {
                        //swap example with medoid
                        Example ex = c.getCluster().getExample(e);
                        c.DeleteExample(ex);
                        Example medoid = clusters.get(a).getRepresentative();
                        clusters.get(a).AddExample(medoid);
                        clusters.get(a).SetRepresentative(ex);
                        //calculate new distortion
                        double newdistortion = 0;
                        for (Cluster x : clusters) {
                            newdistortion += x.distortion();
                        }
                        //swap back if new distortion is larger than old distortion
                        if (distortion < newdistortion) {
                            clusters.get(a).DeleteExample(medoid);
                            c.AddExample(e, ex);
                            clusters.get(a).SetRepresentative(medoid);
                        //change is true, keep iterating
                        } else {
                            change = true;
                        }
                    }
                }
            }
        }
        //getting each representative medoid from all the clusters and add to output set
        Set medoids = new Set(clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames());
        for (Cluster c : clusters) {
            medoids.addExample(c.getRepresentative());
        }
        return medoids;
    }
}
