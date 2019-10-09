/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 * Cluster class has a representative example and set of points in it cluster
 * used for simplifying clustering algorithms
 * @author Kevin
 */
public class Cluster {
    //representative point
    private Example representative;
    //distance metric used to calculate distortion for C Medoids
    private EuclideanSquared euclidean;
    //cluster of points under representative point
    private Set cluster;
    /**
     * constructor takes in a representative example, and set attributes to make a new
     * set for that example
     * @param rep
     * @param num_attributes
     * @param num_classes
     * @param class_names 
     */
    Cluster(Example rep, int num_attributes, int num_classes, String[] class_names) {
        this.representative = rep;
        this.cluster = new Set(num_attributes, num_classes, class_names);
    }
    /**
     * calculates distortion for C medoids
     * @return 
     */
    public double distortion(){
        double distortion = 0;
        for (Example ex : cluster){
            distortion += euclidean.dist(ex, representative);
        }
        return distortion;
    }
    //getters and setters for cluster and representative
    public void AddExample(Example example){ cluster.addExample(example); }
    public void DeleteExample(Example example){ cluster.delExample(example); };
    public void SetRepresentative(Example example){ representative = example; }
    public Example getRepresentative(){ return this.representative; }
    public Set getCluster(){ return this.cluster; }
    public void clearCluster(){ this.cluster.clearSet(); }
}
