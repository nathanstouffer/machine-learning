/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package evaluatelearner;

import datastorage.Set;
import neuralnets.layer.Vector;

/**
 *
 * @author natha
 */
public class AutoEncoderEvaluator implements IEvaluator {
    
    private double mse;
    private double mae;
    private double me;
    
    public AutoEncoderEvaluator(Set predicted, Set actual) {
        // An AutoEncoder looks only at the attributes of examples
        // we have normalized the attributes in preprocessing,
        // so there is no need to compute z-scores
        
        int num_examples = actual.getNumExamples();
        
        for (int i = 0; i < num_examples; i++) {
            if (i % (num_examples / 5) == 0) {
                System.out.println(predicted.getExample(i).toString());
                System.out.println(actual.getExample(i).toString());
                System.out.println();
            }
        }     
        
        Vector[] differences = new Vector[num_examples];
        // compute the difference of each attribute in each example
        for (int i = 0; i < num_examples; i++) {
            differences[i] = new Vector(actual.getNumAttributes());
            // iterate through attributes
            for (int j = 0; j < actual.getNumAttributes(); j++) {
                double pred = predicted.getExample(i).getAttributes().get(j);
                double act = actual.getExample(i).getAttributes().get(j);
                double diff = pred - act;
                differences[i].set(j, diff);
            }
        }
        
        double[] mses = new double[num_examples];
        double[] maes = new double[num_examples];
        double[] mes = new double[num_examples]; 
        
        int num_attr = actual.getNumAttributes();
        // populate metric arrays
        for (int i = 0; i < num_examples; i++) {
            // compute each metric for each example
            for (int j = 0; j < differences[0].getLength(); j++) {
                double difference = differences[i].get(j);
                mes[i] += difference;
                maes[i] += Math.abs(difference);
                mses[i] += Math.pow(difference, 2);
            }
            mes[i] /= num_attr;
            maes[i] /= num_attr;
            mses[i] /= num_attr;
        }
        
        // compute average of each metric
        this.me = 0;
        this.mae = 0;
        this.mse = 0;
        for (int i = 0; i < num_examples; i++) {
            this.mse += mses[i];
            this.mae += maes[i];
            this.me += mes[i];
        }
        //Take the average
        mse /= num_examples;
        mae /= num_examples;
        me /= num_examples;
    }

    @Override
    public double getAccuracy() { return -1.0; } // unimplemented 

    @Override
    public double getMSE() { return mse; }

    @Override
    public double getMAE() { return mae; }

    @Override
    public double getME() { return me; }
    
}
