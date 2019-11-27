/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package poptrain;

import datastorage.Set;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.util.Random;
import neuralnets.MLP;
import neuralnets.layer.Vector;
import java.util.Arrays;

/**
 *
 * @author natha
 */
public class Particle implements Comparable<Particle> {
    
    /**
     * upper and lower bounds for omega.
     * omega is the multiplier for the current velocity
     * when computing momentum
     */
    private final double WUPPER = 2.0;
    private final double WLOWER = 0.4;
    
    /**
     * bound on the magnitude of a component of the velocity vector
     */
    private final double VELCLAMP = 0.0001;
    
    /**
     * current network, the weights will be updated according
     * to the Vector pos as the velocity update rule is applied
     */
    private final MLP network;
    
    /**
     * Final constant multipliers for past experience and group experience.
     * These will be used in the velocity update rule
     */
    private final double pc;
    private final double gc;
    
    /**
     * private double to store the fitness of the network
     */
    private double fitness;
    
    /**
     * Vector describing the current network in Vector format
     * for ease of manipulation
     */
    private Vector pos;
    
    /**
     * private vector to store previous velocity of the particle
     * This is used in the Velocity update rule
     */
    private Vector vel;
    
    /**
     * Vector to store the vector form of the particle's best performing
     * network
     */
    private Particle pBest;
    
    /**
     * Static Vector to store the vector form of the group's current 
     * best performing network
     */
    private static Particle gBest;
    
    /**
     * These are the testing examples used for evaluating the 
     * fitness of the network
     */
    private static Set data;
    
    // random number generator
    private Random rand;
    
    /**
     * Protected constructor to build a particle that
     * moves around in the cost space, searching for
     * the optimal configuration
     * 
     * @param pc
     * @param gc
     * @param network 
     */
    protected Particle(double pc, double gc, MLP network) {
        // set global variables
        this.network = network;
        this.pos = network.toVec();
        this.computeFitness();
        this.pc = pc;
        this.gc = gc;
        this.pBest = new Particle(network);
        this.vel = new Vector(this.pos.getLength());
        this.vel.randPopulate(-1*this.VELCLAMP, this.VELCLAMP);
        // set generation best variables to null
        Particle.gBest = null;
        this.rand = new Random();
    }
    
    /**
     * Protected constructor to store a current optimal 
     * configuration of a network
     * 
     * @param network 
     */
    protected Particle(MLP network) {
        this.network = network.clone(); // clone so seperate memory addresses
        this.pos = network.toVec();
        this.computeFitness();
        // dummy values for final variable
        this.pc = 0;
        this.gc = 0;
    }
    
    /**
     * public method to return the which of this and p 
     * are more fit
     * 
     * 1 is returned if this is more fit than p
     * 0 is returned if the fitness are equal
     * -1 is return if this is less fit than p
     * @param p
     * @return 
     */
    @Override
    public int compareTo(Particle p) {
        if (this.fitness > p.getFitness()) { return 1; }
        else if (this.fitness == p.getFitness()) { return 0; }
        else { return -1; }
    }
    
    /**
     * protected method to update the position via the velocity
     * update rule.
     * 
     * omega is based on the percentage of iterations that have
     * been run in the algorithm
     * 
     * @param perc percentage of iterations completed
     */
    protected void updatePos(double perc) {
        // check if necessary variables exist
        if (Particle.gBest == null) { System.err.println("global best must be computed"); }
        else {
            // compute new velocity
            this.vel = this.computeVel(perc);

            // update position
            this.pos.plusEquals(this.vel);
            this.network.setWeights(this.pos);

            // compute new fitness
            this.computeFitness();

            // compare to past best and act appropriately
            if (this.compareTo(this.pBest) >= 0) { this.pBest.setPos(this.pos); }
        }
    }
    
    
    /**
     * private method to compute  and set the fitness of a network 
     */
    public void computeFitness() {
        if (Particle.data == null) { System.err.println("no data to test on"); }
        else {   
            double[] results = this.network.test(Particle.data);
            //System.out.println(Arrays.toString(results));
            if (Particle.data.getNumClasses() == -1) {
                // data set is regression
                RegressionEvaluator eval = new RegressionEvaluator(results, Particle.data);
                this.fitness = -1 * eval.getMSE(); // multiply by -1 because of compareTo logic
            }
            else {
                // data set is classification
                ClassificationEvaluator eval = new ClassificationEvaluator(results, Particle.data);
                this.fitness = eval.getAccuracy();
            }
        }
    }
    
    /**
     * private method to return the velocity for this step
     * @param perc
     * @return 
     */
    private Vector computeVel(double perc) {
        // compute inertia component
        double omega = this.WUPPER - (perc * (this.WUPPER - this.WLOWER));
        Vector inertia = this.vel.times(omega);
        
        // compute memory-based component
        Vector pdiff = this.pBest.getPos().minus(this.pos);
        double pr = this.rand.nextDouble();
        pdiff.timesEquals(this.pc);         // multiply by past multiplier
        pdiff.timesEquals(pr);              // multiply by random value
        
        // compute social-based component
        Vector gdiff = Particle.gBest.getPos().minus(this.pos);
        double gr = this.rand.nextDouble();
        gdiff.timesEquals(this.gc);         // multiply by global multiplier
        gdiff.timesEquals(gr);              // multiply by random value
        
        // compute velocity by summing components
        Vector temp = pdiff.plus(gdiff);
        temp.plusEquals(inertia);
        
        // clamp velocity
        for (int t = 0; t < temp.getLength(); t++) {
            if (temp.get(t) > this.VELCLAMP) { temp.set(t, this.VELCLAMP); }
            else if (temp.get(t) < -1*this.VELCLAMP) { temp.set(t, -1*this.VELCLAMP); }
        }
        
        //System.out.println(temp.toString());
        
        return temp;
    }
    
    /**
     * public method to set the position of a particle
     * this also updates the associated network and fitness
     * @param temp 
     */
    protected void setPos(Vector temp) { 
        this.pos = temp.clone();
        this.network.setWeights(this.pos);
        this.computeFitness();
    }
    
    protected static void setTrainingExamples(Set temp) { data = temp; }
    protected static void setGenBest(Particle best) { Particle.gBest = best; Particle.gBest.computeFitness(); }
    protected static Particle getGenBest() { return Particle.gBest; }
    
    protected double getFitness() { return this.fitness; }
    protected Vector getPos() { return this.pos; }
    protected MLP getNetwork() { return this.network; }
    
}
