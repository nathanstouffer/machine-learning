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

    private IDistMetric metric;
    
    private Set cluster;
    
    Cluster(Example rep, IDistMetric metric, int num_attributes, int num_classes, String[] class_names) {
        this.representative = rep;
        this.metric = metric;
        this.cluster = new Set(num_attributes, num_classes, class_names);
    }
    public double distortion(){
        double distortion = 0;
        for (Example ex : cluster){
            distortion += metric.dist(ex, representative);
        }
        return distortion;
    }
    
    public void clusterAdd(Example example){ cluster.addExample(example); }
    public void clusterDelete(Example example){ cluster.delExample(example); };
    public void representativeChange(Example example){ representative = example; }
}
