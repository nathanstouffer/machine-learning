/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

/**
 *
 * @author natha
 */
public class SAE {
    
    private final int num_encoders;
    private final double penalty;
    private final double learning_rate;
    private final boolean noise;
    
    public SAE(int num_encoders, double penalty, double learning_rate) {
        this.num_encoders = num_encoders;
        this.penalty = penalty;
        this.learning_rate = learning_rate;
        this.noise = false;
    }
    
}
