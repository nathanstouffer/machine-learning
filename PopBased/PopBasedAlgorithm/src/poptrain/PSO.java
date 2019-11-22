/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package poptrain;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
import neuralnets.INeuralNet;
import neuralnets.MLP;
import neuralnets.layer.Vector;

/**
 * Class to train a MLP feed-forward network using Particle
 * Swarm Optimization. 
 * 
 * @author natha
 */
public class PSO implements IPopTrain {
    
    public final double cp;
    public final double cg;
    
    private final int pop_size;
    
    private final int max_iter;
    
    
    /**
     * Public constructor to set up a population of particles
     * to be trained using Particle Swarm Optimization
     * 
     * the array topology is expected to have the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth 
     * hidden layer
     * 
     * @param cog_mult
     * @param soc_mult
     * @param topology { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * @param pop_size
     * @param max_iter 
     */
    public PSO(double cog_mult, double soc_mult, int[] topology, 
            int pop_size, int max_iter, SimilarityMatrix[] sim) {
        // set global variables
        this.cp = cog_mult;
        this.cg = soc_mult;
        this.pop_size = pop_size;
        this.max_iter = max_iter;
        
        // initialize members of pop
    }
    
    // dummy constructor for testing
    public PSO(int[] topology, SimilarityMatrix[] sim) {
        this.cp = 0;
        this.cg = 0;
        this.pop_size = 0;
        this.max_iter = 0;
        
        MLP mlp = new MLP(topology, sim);
        mlp.randPopWeights();
        System.out.println(mlp.getLayer(0).getWeights().toString());
        Vector vec = mlp.toVec();
        Vector dummy = new Vector(vec.getLength());
        for (int v = 0; v < dummy.getLength(); v++) { dummy.set(v, 1); }
        
        vec.plusEquals(dummy);
        mlp.setWeights(vec);
        System.out.println(mlp.getLayer(0).getWeights().toString());
    }

    @Override
    public MLP train(Set training) {
        return new MLP(new int[] { 0 }, new SimilarityMatrix[] { });
    }
    
}
