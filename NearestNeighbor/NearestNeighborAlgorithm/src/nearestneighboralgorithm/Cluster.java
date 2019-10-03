/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 *
 * @author Kevin
 */
public class Cluster {

    private Example representative;

    private Set cluster;
    
    private EuclideanSquared euclidean;
    
    Cluster(Example example, int num_attributes, int num_classes, String[] class_names) {
        this.representative = example;
        this.cluster = new Set(num_attributes, num_classes, class_names);
    }
    public double distortion(){
        double distortion = 0;
        for (Example ex : cluster){
            distortion += euclidean.dist(ex, representative);
        }
        return distortion;
    }
    
    public void clusterAdd(Example example){ cluster.addExample(example); }
    public void clusterDelete(Example example){ cluster.delExample(example); };
    public void representativeChange(Example example){ representative = example; }
}
