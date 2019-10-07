/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

import java.util.ArrayList; 
import java.util.Random;
import java.lang.Double;
/**
 *
 * @author erick
 */
public class CMeans implements IDataReducer{

    Random rd = new Random();
    ArrayList<Cluster> clusterList;
    Set reduced;
    ArrayList<Example> original;
    ArrayList<Example> clone;
    Set oldcenters;
    int c;

    private IDistMetric metric;
    private KNNClassifier learner;
    private Set validation_set;    
    
    CMeans(int k, IDistMetric metric, Set validation_set){
        this.c = k;
        this.metric = metric;
        this.learner = new KNNClassifier();
        this.learner.setK(k);
        this.learner.setDistMetric(metric);
        
        this.validation_set = validation_set;
    }
   
   public Set reduce(Set original){
       int maxIter = 1000;
       int i = 0;
       initializeClusters();
       populateClusters();
       while (i <= maxIter && oldcenters != reduced){
           oldcenters = reduced.clone();
           updateCenters();
           maxIter++;
       }
       return(reduced);
   }
   
   public void initializeClusters(){
       for(int i = 0; i < c; i++){
           reduced.addExample(clone.get(rd.nextInt(clone.size())));
       }
   }
   
   public void populateClusters(){
       for(Example ex : original){
           double min = Double.MAX_VALUE;
           Cluster clusterToAddTo;
           for (Example center:reduced){
               double dist = metric.dist(ex, center);
               if (dist < min){
                   min = dist;
                   clusterToAddTo = 
               }
           }
       }
   }
   
   public void updateCenters(){
       for (int j = 0; j < reduced.getNumExamples(); j++){
           Example center = reduced.getExample(j);
           ArrayList<Double> newAttributes = new ArrayList<Double>(center.getAttributes().size());
           for (int i = 0; i < center.getAttributes().size(); i++){
               for (Example ex : clone){
                   ArrayList<Double> exAttr = ex.getAttributes();
                   double currentsum = newAttributes.get(i) + exAttr.get(i);
                   newAttributes.set(i, currentsum);
               }
               double average = newAttributes.get(i)/clone.size();
               newAttributes.set(i, average);
           }
           Example newCenter = new Example(newAttributes);
           reduced.replaceExample(j, newCenter);
       }
    }
}

