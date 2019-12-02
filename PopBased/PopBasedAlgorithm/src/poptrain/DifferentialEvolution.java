/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package poptrain;

import datastorage.Set;
import datastorage.SimilarityMatrix;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.util.Random;
import neuralnets.MLP;
import neuralnets.layer.Vector;

/**
 * Utilized Differential Evolution to train an MLP. Preforms cycles of mutation, 
 * crossover and replacement. Uses uniform crossover, random selection and 
 * single difference vector mutation.
 * @author Kevin
 */
public class DifferentialEvolution implements IPopTrain {
    
    /**
     * These parameters define the behavior of the algorithm. Topology describes
     * the MLP's dimensions, layers, and general topology. Population size 
     * can be tuned to affect the search capabilities. Crossover rate will 
     * determine the rate at which crossover occurs, mutation rate will
     * determine the rate at which mutation occurs. Max generations will be used
     * to end the training.
     */
    private final int[] topology;
    /*The array topology is expected to have the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth hidden layer */
    private final int population_size;
    private final double crossover_rate;
    private final double mutation_rate;
    private final int max_generations;
    
    /**
     * These arrays contain the vectors that we will be manipulating
     * to search through the hyperspace.
     */
    private Vector[] population;
    private double[] population_fitnesses;
    
    /**
     * The set of data that will be used to evaluate the fitness.
     */
    private Set data;
    
    /**
     * The similarity matrices used by various networks. Save to pass into
     * construction.
     */
    private final SimilarityMatrix[] sim;
    
    /**
     * Best individual
     */
    private Vector all_time_best;
    private double all_time_best_fitness;
    
    /**
     * The random number generator used during selection.
     */
    private final Random rand;

    /**
     *
     * @param topology The topology of the MLP network that is being trained.
     * The array topology is expected to have the form { num_hl, num_hn_1,
     * num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * @param population_size Size of population used in algorithm
     * @param crossover_rate The rate (between 0 and 0.5) at which (uniform)
     * crossover occurs.
     * @param mutation_rate The rate (between 0 and 1) at which mutation occurs.
     * @param max_generations Add a maximum number of iterations to the training
     * process.
     * @param sim The similarity matrices used for handling categorical
     * features.
     */
    DifferentialEvolution(int[] topology, int population_size, double crossover_rate, double mutation_rate, int max_generations, SimilarityMatrix[] sim) {
        this.topology = topology;
        this.population_size = population_size;
        this.crossover_rate = crossover_rate;
        this.mutation_rate = mutation_rate;
        this.max_generations = max_generations;
        this.sim = sim;
        this.rand = new Random();
        this.all_time_best_fitness = 0;
    }
    /**
     * Trains an MLP using Differential Evolution
     * @param training Training data set
     */
    @Override
    public void train(Set training){
        //iterate for a specified number of generations
        int generation = 0;
        while(generation < max_generations){
            //each generation when iterations is equal to pop size 
            for(int i = 0; i < population_size; i++){
                //randomly select target vector and compute fitness
                int target_index = rand.nextInt(population_size);
                Vector target_vector = population[target_index];
                double target_fitness = computeFitness(target_vector);
                //create trial vector and compute fitness
                Vector trial_vector = mutate();
                trial_vector = cross(target_vector, trial_vector);
                double trial_fitness = computeFitness(trial_vector);
                //if trial fitness is greater than target fitness, replaces
                //target with trial in general population
                if(target_fitness < trial_fitness){
                    population[target_index] = trial_vector;
                }
            }
            generation++;  
        }
    }
    /**
     * Returns the best individual in the population
     * @return Best MLP in population
     */
    @Override
    public MLP getBest() {
        //calculate fitness for all members of population
        for(int i = 0; i < population.length; i++) { 
            population_fitnesses[i] = computeFitness(population[i]);
        }
        //determine most fit individual in population 
        for(int i = 0; i < population.length; i++){
            if(population_fitnesses[i] > all_time_best_fitness){
                all_time_best_fitness = population_fitnesses[i];
                all_time_best = population[i];
            }
        }
        //create MLP from most fit individual
        MLP network = new MLP(topology, sim);
        network.setWeights(all_time_best);
        return network;
    }
    /**
     * Creates trial vector with a single difference vector 
     * @return trial vector
     */
    private Vector mutate(){
        //randomly selecting vectors from population
        Vector[] trial_vector_components = {population[rand.nextInt(population_size)],population[rand.nextInt(population_size)],population[rand.nextInt(population_size)]};
        //compute x1 + B(x2 - x3)
        Vector trial_vector = trial_vector_components[1].minus(trial_vector_components[2]);
        trial_vector = trial_vector.times(mutation_rate);
        trial_vector = trial_vector_components[0].plus(trial_vector);
        return trial_vector;
    }
    /**
     * Crosses target and trial vector based on crossover rate using uniform 
     * using uniform crossover
     * @param target Target vector
     * @param trial Trial vector
     * @return crossed trial vector
     */
    private Vector cross(Vector target, Vector trial) {
        // Clone the vectors
        target = target.clone(); trial = trial.clone();
        // Iterate through each vector value, testing if the values will be
        // crossed
        for(int i = 0; i < target.getLength(); i++) {
            // Generate a random double to see if crossover will occur
            if(rand.nextDouble() <= crossover_rate) {
                // Swap values
                double temp = target.get(i);
                target.set(i, trial.get(i));
                trial.set(i, temp);
            } 
        }
        Vector crossed = trial;
        return crossed;
    }

    private void initializePopulation() {
        // Initialize the arrays that hold our populations
        population = new Vector[population_size];

        // Initialize all time best info
        all_time_best = null;
        all_time_best_fitness = Double.MIN_VALUE;

        // Fill the initial population
        for (int i = 0; i < population.length; i++) {
            MLP network = new MLP(topology, sim); // Create a network with a 
            // given topology.
            network.randPopWeights(); // Randomly populate the weights in the 
            // network.
            population[i] = network.toVec(); // Convert the netowrk to a vector 
            // and store it in our array.
        }
    }

    private double computeFitness(Vector individual) {

        // Create an MLP from the individual
        MLP network = new MLP(topology, sim);
        network.setWeights(individual);

        // Test the network using the training data in "data"
        double[] results = network.test(data);
        if (data.getNumClasses() == -1) {
            // data set is regression
            RegressionEvaluator eval = new RegressionEvaluator(results, data);
            return eval.getMSE() * -1; // Multiply by negative 1 because we 
            // assume good fitness is greater.
        } else {
            // data set is classification
            ClassificationEvaluator eval = new ClassificationEvaluator(results, data);
            return eval.getAccuracy();
        }
    }

}
