/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import datastorage.*;
import networklayer.*;

/**
 *
 * @author natha
 */
public interface INeuralNet {
    
    public void train(Set training_set);
    public double[] test (Set testing_set);
    public double predict(Example ex);
    public Vector[] genLayerOutputs(Example ex);
    
}
