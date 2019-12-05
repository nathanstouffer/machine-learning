/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package poptrain;

import datastorage.Example;
import datastorage.Set;
import datastorage.SimilarityMatrix;
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
    private final double VELCLAMP = 1;
    
    /**
     * Final constant multipliers for past experience and group experience.
     * These will be used in the velocity update rule
     */
    private final double pc;
    private final double gc;
    
    /**
     * private integer array to store the topology of the network
 that this particle represents
     */
    private int[] topology;
    
    /**
     * private array of similarity matrices to be used when constructing
     * the MLP that the pos Vector represents
     */
    private SimilarityMatrix[] sim;
    
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
     * Static Particle to store the vector form of the group's current 
     * best performing network
     */
    private static Particle gBest;
    
    /**
     * static integer to store the index of the generation best
     */
    private static int gBestIndx;
    
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
    protected Particle(double pc, double gc, int[] topology, SimilarityMatrix[] sim) {
        // set global variables
        this.topology = topology;
        this.sim = sim;
        // randomly initialize the network
        MLP network = new MLP(this.topology, this.sim);
        network.randPopWeights();
        // set global variables
        this.pos = network.toVec();
        this.fitness = this.computeFitness();
        this.pc = pc;
        this.gc = gc;
        this.pBest = new Particle(this.topology, this.sim, this.pos);
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
    protected Particle(int[] topology, SimilarityMatrix[] sim, Vector pos) {
        // set global variables
        this.topology = topology;
        this.sim = sim;
        this.pos = pos.clone();
        this.fitness = this.computeFitness();
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

            // compute new fitness
            this.fitness = this.computeFitness();

            // compare to past best and act appropriately
            if (this.compareTo(this.pBest) >= 0) { this.pBest.setPos(this.pos); }
        }
    }
    
    
    /**
     * private method to compute  and set the fitness of a network 
     */
    public double computeFitness() {
        if (Particle.data == null) { System.err.println("no data to test on"); return 0.0; }
        else {   
            //System.out.println(Arrays.toString(results));
            if (Particle.data.getNumClasses() == -1) {
                // data set is regression
                return this.maeFitness();
            }
            else {
                // data set is classification
                return this.accFitness();
                //return this.maeFitness();
            }
        }
    }
    
    /**
     * private method to compute the accuracy of the particle. This method
     * should only be called for classification datasets
     */
    private double accFitness() {
        MLP network = new MLP(this.topology, this.sim);
        network.setWeights(this.pos);
        double[] results = network.test(Particle.data);
        ClassificationEvaluator eval = new ClassificationEvaluator(results, Particle.data);
        return eval.getAccuracy();
    }
    
    private double maeFitness() {
        // construct temporary network
        MLP network = new MLP(this.topology, this.sim);
        network.setWeights(this.pos);
        if (Particle.data.getNumClasses() == -1) {
            double[] results = network.test(Particle.data);
            RegressionEvaluator eval = new RegressionEvaluator(results, Particle.data);
            return -1 * eval.getMAE(); // multiply by -1 because of compareTo logic
        }
        else {
            // data set is classification
            double mae = 0.0;
            int num_layers = network.getLayerDim().length;
            // iterate through examples
            for (int d = 0; d < Particle.data.getNumExamples(); d++) {
                // current example
                Example ex = Particle.data.getExample(d);
                int actual = (int)ex.getValue();
                // class probabilities for example
                Vector output = network.genLayerOutputs(ex)[num_layers];
                // create and populate target vector
                Vector target = new Vector(output.getLength());
                for (int t = 0; t < target.getLength(); t++) {
                    if (t == actual) { target.set(t, 1.0); }
                    else { target.set(t, 0.0); }
                }
                // compute summed mae for each class probability
                double diff = 0.0;
                for (int o = 0; o < output.getLength(); o++) {
                    diff += Math.abs(output.get(o) - target.get(o));
                }
                mae += diff;
            }
            // average the ma
            mae /= Particle.data.getNumExamples();
            // set fitness
            return -1 * mae;
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
        
        //System.out.println(network.toString());
        
        return temp;
    }
    
    /**
     * protected method to compute the distance from this vector to the
     * generation best vector
     * @return 
     */
    protected double distToGenBest() {
        // current and gen best positions
        Vector curr = this.pos;
        Vector best = Particle.gBest.getNetwork().toVec();
        Vector diff = curr.minus(best);
        double dist = 0.0;
        // sum squared differences
        for (int d = 0; d < diff.getLength(); d++) { 
            dist += Math.abs(diff.get(d));
        }
        //dist = Math.sqrt(dist);
        return dist;
    }
    
    /**
     * public method to set the position of a particle
     * this also updates the associated network and fitness
     * @param temp 
     */
    protected void setPos(Vector temp) { 
        this.pos = temp.clone();
        this.fitness = this.computeFitness();
    }
    
    protected MLP getNetwork() { 
        // construct network
        MLP network = new MLP(this.topology, this.sim);
        network.setWeights(this.pos);
        return network; 
    }
    
    /**
     * protected method to set the generations current best performing
     * particle
     * @param best 
     */
    protected static void setGenBest(int index, Particle best) { 
        Particle.gBestIndx = index;
        Particle.gBest = best;
    }
    
    protected static void setTrainingExamples(Set temp) { Particle.data = temp; }
    protected static Particle getGenBest() { return Particle.gBest; }
    protected static int getGBestIndex() { return Particle.gBestIndx; }
            
    protected double getFitness() { return this.fitness; }
    protected Vector getPos() { return this.pos; }
    
}
