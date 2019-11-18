/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package evaluatelearner;

import datastorage.Example;
import java.util.ArrayList;
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
    
    public AutoEncoderEvaluator(Vector[] predicted, Vector[] actual) {
    //public AutoEncoderEvaluator(Set predicted, Set actual) {
        // An AutoEncoder looks only at the attributes of examples
        // we have normalized the attributes in preprocessing,
        // so there is no need to compute z-scores
        
        int num_examples = actual.length;
        
        for (int i = 0; i < num_examples; i++) {
            if (i % (num_examples / 5) == 0) {
                /*System.out.println("------------ ACTUAL ------------");
                System.out.println(actual.getExample(i).toString());
                System.out.println("------------ PREDICTED ------------");
                System.out.println(predicted.getExample(i).toString());*/
                System.out.println("------------ DIFFERENCE ------------");
                ArrayList<Double> differences = new ArrayList<Double>();
                for (int j = 0; j < predicted[0].getLength(); j++) {
                    double pred = predicted[i].get(j);
                    double act = actual[i].get(j);
                    differences.add(pred - act);
                }
                Example diff = new Example(-1.0, differences);
                System.out.println(diff);
                System.out.println();
            }
        }     
        
        Vector[] differences = new Vector[num_examples];
        // compute the difference of each attribute in each example
        for (int i = 0; i < num_examples; i++) {
            differences[i] = new Vector(actual[0].getLength());
            // iterate through attributes
            for (int j = 0; j < differences[0].getLength(); j++) {
                double pred = predicted[i].get(j);
                double act = actual[i].get(j);
                double diff = pred - act;
                differences[i].set(j, diff);
            }
        }
        
        double[] mses = new double[num_examples];
        double[] maes = new double[num_examples];
        double[] mes = new double[num_examples]; 
        
        int num_attr = actual[0].getLength();
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
