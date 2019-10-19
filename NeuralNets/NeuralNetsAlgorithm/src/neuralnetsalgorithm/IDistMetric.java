/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetsalgorithm;

/**
 * Interface that requires realizing classes to provide a 
 * method called dist. This method should compute a distance
 * between two elements of a data set using a type of
 * distance metric specified by the realizing class
 * 
 * @author natha
 */
public interface IDistMetric {

    public double dist(Example ex1, Example ex2);

}
