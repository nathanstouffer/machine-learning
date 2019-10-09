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

    Random rd = new Random();
    ArrayList<Cluster> clusterList;
    ArrayList<Example> oldcenters;
    int c;
    private IDistMetric metric;
    
    /**
     * Constructor to initialize global variables
     * @param k
     * @param metric 
     */
    CMeans(int k, IDistMetric metric){
        this.c = k;
        this.metric = metric;
        this.oldcenters = new ArrayList<>();
        this.clusterList = new ArrayList<>();
    }
   
    /**
     * Method to reduce the size of a Set for use with K nearest neighbors
     * @param original
     * @return 
     */
    @Override
   public Set reduce(Set original){
       oldcenters = new ArrayList<>();
       clusterList = new ArrayList<>();
       int maxIter = 1000;
       int i = 0;
       initializeClusters(original);
       populateClusters(original);
       boolean different = true;
       while (i <= maxIter && different){               //main loop for updating clusters
           for (Cluster cluster : clusterList){         //creates an ArrayList to compare changes made each iteration
               oldcenters.add(cluster.getRepresentative());
           }
           updateCenters(original);
           populateClusters(original);
           i++;
           for (int j = 0; j < clusterList.size(); j++){        //loops through all clusters amd checks if they have changed
               if (clusterList.get(j).getRepresentative().getAttributes().equals(oldcenters.get(j).getAttributes())){
                   different = false;
               }
           }
       }
       Set reduced = new Set(original.getNumAttributes(), original.getNumClasses(), original.getClassNames());  //adds cluster centers to a set to return
       for (Cluster cluster : clusterList){
           reduced.addExample(cluster.getRepresentative());
       }
       return(reduced);
   }
   
   /**
    * Method for randomly initializing random points throughout the data set as new cluster centers
    * @param original 
    */
   public void initializeClusters(Set original){
       for (int i = 0; i < c; i++){
           Example center = original.getExample(rd.nextInt(original.getNumExamples()));
           Cluster cluster = new Cluster(center, metric, original.getNumAttributes(), original.getNumClasses(), original.getClassNames());
           clusterList.add(cluster);
       }
   }
   
   /**
    * Method for populating clusters with examples from the original Set
    * @param original 
    */
   public void populateClusters(Set original){
       for (Cluster cluster : clusterList){             //resets clusters so they can be repopulated if already filled
           cluster.clearCluster();
       }
       for(Example ex : original){
           double min = Double.MAX_VALUE;
           Cluster closestCluster = clusterList.get(0);
           for (Cluster cluster : clusterList){                 //finds closest cluster to Example ex
               double dist = metric.dist(ex, cluster.getRepresentative());
               if (dist < min){
                   min = dist;
                   closestCluster = cluster;
               }
           }
           closestCluster.AddExample(ex);
       }
   }
   
   /**
    * Method for updating cluster centers by taking average value of points in each cluster
    * @param original 
    */
   public void updateCenters(Set original){
       for (Cluster cluster : clusterList){
           Example center = cluster.getRepresentative();
           ArrayList<Double> newAttributes = new ArrayList<>(center.getAttributes().size());    //makes an ArrayList to calculate averages
           for (int i = 0; i < center.getAttributes().size(); i++) { //Initialize Arraylist with 0s
               newAttributes.add(0.0);
           }
           for (int i = 0; i < center.getAttributes().size(); i++){
               for (Example ex : original){                                                     //gets the average of each attribute in the cluster
                   ArrayList<Double> exAttr = ex.getAttributes();
                   double currentsum = newAttributes.get(i) + exAttr.get(i);
                   newAttributes.set(i, currentsum);
               }
               double average = newAttributes.get(i)/original.getNumExamples();
               newAttributes.set(i, average);
           }
           Example newCenter = new Example(newAttributes);              //puts the averages into an example to save as the new center
           cluster.SetRepresentative(newCenter);
       }
    }
}

