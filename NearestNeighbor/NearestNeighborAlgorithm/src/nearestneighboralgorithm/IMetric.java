/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 *
 * @author natha
 */
public interface IMetric {
    
    public double dist(Example ex1, Example ex2);

}
