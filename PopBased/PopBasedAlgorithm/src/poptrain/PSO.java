/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package poptrain;

import datastorage.Set;
import datastorage.SimilarityMatrix;
import neuralnets.MLP;

/**
 * Class to train a MLP feed-forward network using Particle
 * Swarm Optimization. 
 * 
 * @author natha
 */
public class PSO implements IPopTrain {
    
    /**
     * Final constant multipliers for past experience and group experience.
     * These will be used in the velocity update rule
     */
    private final double pc;
    private final double gc;
    
    /**
     * This is the population size that we will work with
     * It is a tuneable parameter
     */
    private final int pop_size;
    
    /**
     * The number of maximum iterations that the swarm
     * will run through
     */
    private final int max_iter;
    
    /**
     * An array that describes the topology of the networks
     * to be created in this swarm
     * 
     * This array is of the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth 
     * hidden layer
     */
    private final int[] topology;
    
    /**
     * private variable to store the similarity matrices for the
     * data set that PSO is ran on
     */
    private final SimilarityMatrix[] sim;
    
    /**
     * private array to store the population of individuals in 
     * the swarm
     */
    private Particle[] pop;
    
    /**
     * private variable to store the particle with the best
     * performance, regardless of time
     */
    private Particle best;
    
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
     * @param sim 
     */
    public PSO(double cog_mult, double soc_mult, int[] topology, 
            int pop_size, int max_iter, SimilarityMatrix[] sim) {
        // set global variables
        this.pc = cog_mult;
        this.gc = soc_mult;
        this.pop_size = pop_size;
        this.pop = new Particle[pop_size];
        this.max_iter = max_iter;
        this.topology = topology;
        this.sim = sim;
    }

    @Override
    public void train(Set training) {
        // set training examples
        Particle.setTrainingExamples(training);
        // initialize population
        this.initializePop();
        
        double avg = 0.0;
        boolean converged = false;
        // run for a number of iterations
        for (int iter = 0; iter < this.max_iter && !converged; iter++) {
            // compute percentage
            double perc = (double)iter / this.max_iter;
            // iterate through population
            for (int p = 0; p < this.pop_size; p++) {
                // update position
                this.pop[p].updatePos(perc);
            }
            
            // set generation best_info
            this.setGenBest();
            
            // output to console
            if(iter % (this.max_iter/100) == 0) { 
                double new_avg = this.printIterInfo(iter);
                if (new_avg == avg) { converged = true; }
                avg = new_avg;
                this.printPop(iter);
            }
        }
    }
    
    /**
     * private method to set the static variable for generation
     * best_info in each particle
     * 
     * this method also updates the most optimal network seen in the swarm so far
     * 
     */
    private void setGenBest() {
        // assume the best_info is the 0th member
        Particle curr_best = this.pop[0];
        // iterate thorugh population
        for (int p = 0; p < this.pop_size; p++) {
            if (this.pop[p].compareTo(curr_best) >= 0) { curr_best = this.pop[p]; }
        }
        // set gen best_info in particle
        Particle.setGenBest(curr_best);
        
        //System.out.println(curr_best.getFitness());
        
        // change best_info member seen so far if necessary
        if (curr_best.compareTo(this.best) >= 0) { 
            this.best.setPos(curr_best.getNetwork().toVec());
        }
    }
    
    /**
     * method to initialize a population of particles (neural networks)
     * at random locations (random weight values)
     * 
     */
    private void initializePop() {
        // initialize members of population
        for (int p = 0; p < this.pop_size; p++) {
            // create network
            MLP network = new MLP(this.topology, this.sim);
            // set weights to random values
            network.randPopWeights();
            // put network in population
            this.pop[p] = new Particle(this.pc, this.gc, network);
        }
        
        // calculate best_info members
        this.best = new Particle(this.pop[0].getNetwork());     // set dummy global best_info
        //System.out.println("-> Original fitness: " + this.best_info.getFitness());
        // set generation best_info
        this.setGenBest();
        
        // print population
        this.printPop(-1);
    }
    
    @Override
    public MLP getBest() { return this.best.getNetwork(); }
    
    /**
     * private method to print information about the current iteration
     * 
     * the average fitness of the population is also computed and returned
     * @param iter 
     */
    private double printIterInfo(int iter) {
        String iter_info = "-> Training PSO iteration: " + iter;
        String best_info = String.format("-> Best fitness so far: %.4f", this.best.getFitness());
        String gen_info = String.format("-> Current generation best fitness: %.4f",
                Particle.getGenBest().getFitness());
        // compute average fitness
        double avg = 0.0;
        for (int p = 0; p < this.pop_size; p++) { avg += this.pop[p].getFitness(); }
        avg /= this.pop_size;
        String avg_info = String.format("-> Current generation avg fitness: %.4f", avg);
        String info = String.format("%-35s%-35s%-47s%s", iter_info, best_info, gen_info, avg_info);
        System.out.println(info);
        return avg;
    }
    
    /**
     * private method to output the current fitness
     * of each member of the population
     */
    private void printPop(int iter) {
        System.out.println("------ POPULATION PERFORMANCE FOR ITERATION " + iter + " ------");
        for (int p = 0; p < this.pop_size; p++) {
            String line = "";
            for (int l = 0; l < 10; l++) {
                double temp_fit = Math.abs(this.pop[p].getFitness());
                if (this.pop[p].getFitness() == Particle.getGenBest().getFitness()) {
                    String num = String.format("\u001B[32m%-5s%.4f\u001B[0m   ", 
                            p+": ", temp_fit);
                    line += num;
                }
                else if (p < this.pop_size) {
                    String num = String.format("%-5s%.4f   ", p+": ", temp_fit);
                    line += num;
                }
                p++;
            }
            p--;
            System.out.println(line);
        }
    }
    
}
