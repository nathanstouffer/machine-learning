/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.ArrayList; 
import java.util.Random;
import java.lang.Integer;
/**
 *
 * @author erick
 */
public class CMeans implements iDataReducer{

    Random rd = new Random();
    ArrayList<Cluster> clusterList;
    ArrayList<Example> reduced;
    ArrayList<Example> original;
    ArrayList<Example> clone;
    ArrayList<Example> oldcenters;
    int c;
    
   public CMeans(){
       
   }
   
   public Set reduce(){
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
           reduced.add(clone[rd.nextInt(clone.size())]);
       }
   }
   
   public void populateClusters(){
       Manhattan distance = new Manhattan();
       for(Example ex : original){
           int min = Integer.MAX_VALUE;
           Cluster clusterToAddTo;
           for (Cluster cluster:clusterList){
               dist = distance.dist(ex, cluster)
           }
       }
   }
   
   public void updateCenters(){
       for (Example center : reduced){
           for (double attribute : center.attr()){
               center.attr[attribute] = 0;
               for (Example ex : clone){
                   center.attr[attribute] += example.attr[attribute]
               }
               center.attr[attribute] = center.attr[attribute]/clone.size()
           }
       }
}

