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
 *
 * @author Kevin
 */
public class CMedoids implements IDataReducer{
    
    //number of medoids to create, gets from ENN
    private int c;
    // metric that the reduciing algorithm should use
    private IDistMetric metric;
    //Euclidean squared object to get distances between points
    private ArrayList<Cluster> medoids = new ArrayList<Cluster>();
    CMedoids(int c, IDistMetric met){
        this.c = c;
        this.metric = met;
    }

    @Override
    public Set reduce(Set orig) {
        Set clone = orig.clone();
        Set output = new Set(clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames());
        
        Iterator<Example> iterOrig = clone.iterator();
        Iterator<Example> iterOut = output.iterator();
        //add c number of random medoids from set
        for(int i = 0; i < c; i++){
            int dist = new Random().nextInt(clone.getNumExamples());
            //medoids.add(new Cluster(clone.getExample(dist), clone.getNumAttributes(), clone.getNumClasses(), clone.getClassNames()));
            clone.delExample(clone.getExample(dist));
        }
        boolean change = true;
        while(change){
            iterOrig = clone.iterator();
            iterOut = output.iterator();
            for(int i = 0; i < clone.getNumExamples(); i++){
                double min = -1;
                Example minEx = null;
                Example origEx = iterOrig.next();
                for(int a = 0; a < output.getNumExamples(); a++){
                    Example outEx = iterOut.next();
                    if(min == -1){
                        min = metric.dist(origEx, outEx);
                        minEx = outEx;
                    } else if(min > metric.dist(origEx, outEx)){
                        min = metric.dist(origEx, outEx);
                        minEx = outEx;
                    }
                }
                //add origEx to min output example
            }
            for(int a = 0; a < output.getNumExamples(); a++){
                for(int b = 0; a < clone.getNumExamples(); b++){
                    
                }
            }
        }
        return output;
    }
    
    
}
