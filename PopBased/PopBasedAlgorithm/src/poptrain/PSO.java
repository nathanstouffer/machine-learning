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
 * This class trains MLP networks using the PSO algorithm. A client of this 
 * class should construct an instance by passing in appropriate values
 * for the algorithm's hyper parameters, and then calling the train method.
 * 
 * After calling the train method, a client can retrieve the best-performing
 * member of the swarm with a getter method.
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
     * private variable to store the particle with the best_idx
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
        int prev_best = this.initializePop();
        
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
                double avg_dist = this.printIterInfo(iter);
                if (avg_dist < 0.1) { converged = true; }
                //prev_best = this.printPop(iter, prev_best);
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
        int curr_best = 0;
        // iterate through population
        for (int p = 0; p < this.pop_size; p++) {
            if (this.pop[p].compareTo(this.pop[curr_best]) >= 0) { curr_best = p; }
        }
        // set gen best_info in particle
        Particle.setGenBest(curr_best, this.pop[curr_best]);
        
        //System.out.println(curr_best.getFitness());
        
        // change best_info member seen so far if necessary
        if (this.pop[curr_best].compareTo(this.best) >= 0) { 
            this.best.setPos(this.pop[curr_best].getNetwork().toVec());
        }
    }
    
    /**
     * method to initialize a population of particles (neural networks)
     * at random locations (random weight values)
     * 
     */
    private int initializePop() {
        // initialize members of population
        for (int p = 0; p < this.pop_size; p++) {
            // create network
            MLP network = new MLP(this.topology, this.sim);
            // set weights to random values
            network.randPopWeights();
            // put network in population
            this.pop[p] = new Particle(this.pc, this.gc, this.topology, this.sim);
        }
        
        // calculate best_info members
        this.best = new Particle(this.topology, this.sim, this.pop[0].getPos());     // set dummy global best_info
        //System.out.println("-> Original fitness: " + this.best_info.getFitness());
        // set generation best_info
        this.setGenBest();
        
        // print population
        // print pop returns the index with the best performance
        //return this.printPop(-1, 0);
        return 0;       // use above line for debugging
    }
    
    @Override
    public MLP getBest() { return this.best.getNetwork(); }
    
    /**
     * private method to print information about the current iteration
     * 
     * the average distance to generation best is computed and returned
     * @param iter 
     */
    private double printIterInfo(int iter) {
        String iter_info = "-> Training PSO iteration: " + iter;
        String best_info = String.format("-> Best fitness so far: %.4f", Math.abs(this.best.getFitness()));
        String gen_info = String.format("-> Current generation best fitness: %.4f",
                Math.abs(Particle.getGenBest().getFitness()));
        // compute average fitness
        double avg_fit = 0.0;
        double avg_dist = 0.0;
        for (int p = 0; p < this.pop_size; p++) { 
            avg_fit += Math.abs(this.pop[p].getFitness());
            avg_dist += this.pop[p].distToGenBest();
        }
        avg_fit /= this.pop_size;
        avg_dist /= this.pop_size;
        String avg_info = String.format("-> Current generation avg fitness: %.4f", avg_fit);
        String info = String.format("%-35s%-35s%-47s%s", iter_info, best_info, gen_info, avg_info);
        System.out.println(info);
        return avg_dist;
    }
    
    /**
     * private method to output the current fitness
     * of each member of the population
     */
    private int printPop(int iter, int prev_best) {
        int best_idx = 0;
        System.out.println("------ POPULATION PERFORMANCE FOR ITERATION " + iter + " ------");
        for (int p = 0; p < this.pop_size; p++) {
            String fit_line = "";
            String dist_line = "";
            for (int l = 0; l < 10; l++) {
                if (p < this.pop_size) {
                    double temp_fit = Math.abs(this.pop[p].getFitness());
                    double temp_dist = this.pop[p].distToGenBest();
                    if (p == Particle.getGBestIndex()) {
                        // the best performing individual is printed in green
                        best_idx = p;
                        String num = String.format("\u001B[32m%-5s%.4f\u001B[0m   ", 
                                p+": ", temp_fit);
                        String dist = String.format("\u001B[32m%-5s%6d\u001B[0m   ", 
                                "d: ", 0);
                        fit_line += num;
                        dist_line += dist;
                    }
                    else if (p == prev_best) {
                        // the previous best performing individual is printed in yellow
                        String num = String.format("\u001B[33m%-5s%.4f\u001B[0m   ", 
                                p+": ", temp_fit);
                        String dist = String.format("\u001B[33m%-5s%6.0f\u001B[0m   ", 
                                "d: ", temp_dist);
                        fit_line += num;
                        dist_line += dist;
                    }
                    else {
                        // all other individuals are printed in white
                        String num = String.format("%-5s%.4f   ", p+": ", temp_fit);
                        String dist = String.format("%-5s%6.0f   ", "d: ", temp_dist);
                        fit_line += num;
                        dist_line += dist;
                    }
                }
                p++;
            }
            p--;
            System.out.println(fit_line);
            System.out.println(dist_line);
        }
        return best_idx;
    }
    
}
