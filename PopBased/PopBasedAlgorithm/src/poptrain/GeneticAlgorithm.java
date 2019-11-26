package poptrain;

import datastorage.Set;
import datastorage.SimilarityMatrix;
import evaluatelearner.ClassificationEvaluator;
import evaluatelearner.RegressionEvaluator;
import java.util.Arrays;
import java.util.Random;
import neuralnets.MLP;
import neuralnets.layer.Vector;

/**
 * Utilizes the Genetic Algorithm to train an optimal MLP network. Performs 
 * cycles of selection, crossover, mutation, and replacement as it searches
 * the space of solutions for convergence. 
 * 
 * This specific implementation uses:
 *      -> Rank-based selection
 *      -> Uniform crossover
 *      -> Uniform mutation with real-valued creep
 *      -> Steady-state replacement (1/2 of population is replaced with children)
 * @author andy-
 */
public class GeneticAlgorithm implements IPopTrain {
    
    /**
     * The absolute value of the bounds on the starting weights in the layers.
     */
    private static final double STARTING_WEIGHT_BOUND = 0.0001;
    
    /**
     * Determines the number of children generated each generation by the
     * population size divided by this divisor.
     */
    private static final int NUM_CHILDREN_DIVISOR = 3;
    
    /**
     * These parameters define the behavior of the algorithm. Topology describes
     * the MLP's dimensions, layers, and general topology. Population size 
     * can be tuned to affect the search capabilities. Crossover rate will 
     * determine the rate at which crossover occurs, mutation rate will
     * determine the rate at which mutation occurs. Max generations will be used
     * to end the training.
     */
    private final int[] topology; /*The array topology is expected to have the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth hidden layer */
    private final int population_size;
    private final double crossover_rate;
    private final double mutation_rate;
    private final int max_generations;
    
    /**
     * These arrays contain the vectors that we will be manipulating
     * to search through the hyperspace. This includes the current population,
     * the selected parents of new offspring, and the children.
     * The population rank array stores the relative fitness ranking of the 
     * current population.
     */
    private Vector[] population;
    private double[] population_fitnesses;
    private int[] population_rank;
    private Vector[] parents;
    private Vector[] children;
    
    /**
     * Track the all time best network so that we can return it at the end.
     */
    private Vector all_time_best;
    private double all_time_best_fitness;
    
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
     * The random number generator used during selection, crossover, and 
     * mutation.
     */
    private final Random rand;
    
    /**
     * 
     * @param _topology The topology of the MLP network that is being trained. 
     * The array topology is expected to have the form
     * { num_hl, num_hn_1, num_hn_2, ... num_hn_hl, input_dim, output_dim }
     * where num_hnk denotes the number of hidden nodes in the kth hidden layer.
     * @param _population_size The size of the population used in the algorithm.
     * @param _crossover_rate The rate (between 0 and 0.5) at which (uniform)
     * crossover occurs.
     * @param _mutation_rate The rate (between 0 and 1) at which mutation 
     * occurs. 
     * @param _max_generations Add a maximum number of iterations to the
     * training process.
     * @param _sim The similarity matrices used for handling categorical 
     * features.
     */
    public GeneticAlgorithm(int[] _topology, int _max_generations, int _population_size, 
            double _crossover_rate, double _mutation_rate, SimilarityMatrix[] _sim) {
        topology = _topology;
        population_size = _population_size;
        crossover_rate = _crossover_rate;
        mutation_rate = _mutation_rate;
        max_generations = _max_generations;
        sim = _sim;
        rand = new Random();
    }
    
    /**
     * Trains an MLP using the Genetic Algorithm.
     * @param training_set The set to train the MLP network with
     */
    @Override
    public void train(Set training_set) {
        data = training_set;
        // Initialize the population
        initializePopulation();
        // Begin generational training
        for(int g = 0; g < max_generations; g++){
            // Rank the population based on fitness
            rankPopulation();
            // Select parents and place into the parent array
            selectParents();
            // Cross pairs of parents and place into children array
            crossParents();
            // Mutate all the children
            mutateChildren();
            // Replace (steady state) the children into the general population
            replaceChildren();
            
            // Print out what the current generation is
            if(g % (max_generations/10) == 0) { 
                System.out.println("~~~ GENERATION " + g + " ~~~"); 
                printPopFitness();
                printPopRank();
                printAllTimeBest();
            }
        }
    }
    
    /**
     * Gets the optimal MLP network.
     * @return The best MLP found during the training period.
     */
    @Override
    public MLP getBest() { 
        MLP network = new MLP(topology, sim);
        network.setWeights(all_time_best);
        return network;
    }
    
    /**
     * Computes the fitness of a single individual. Greater value means greater
     * fitness.
     * @param individual The vector representation of an individual MLP
     * @return The fitness of the individual.
     */
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
        }
        else {
            // data set is classification
            ClassificationEvaluator eval = new ClassificationEvaluator(results, data);
            return eval.getAccuracy();
        }
    }
    
    /**
     * Ranks the population, updating the rank array as well as the all time
     * best. Also populates the fitness array.
     */
    private void rankPopulation() {
        // Compute the fitnesses of the current population
        for(int i = 0; i < population.length; i++) { 
            population_fitnesses[i] = computeFitness(population[i]);
        }
        double[] temp_pop_fitnesses = Arrays.copyOf(population_fitnesses, population_fitnesses.length);
        // Fill rank array
        for(int i = 0; i < population.length; i++) {
            int max_index = 0; // Track the best's index
            for(int j = 1; j < population.length; j++) {
                if(temp_pop_fitnesses[j] > temp_pop_fitnesses[max_index]) {
                    max_index = j; // Set the new max index
                }
            }
            // Set the ranking of the max value found, clearing its fitness in
            // the temp array so that it isn't considered again.
            population_rank[max_index] = i;
            temp_pop_fitnesses[max_index] = Double.MIN_VALUE;
            
            
            // Check for an update to the ALL TIME BEST !!!
            if((i == 0) && (population_fitnesses[max_index] > all_time_best_fitness)) {
                all_time_best = population[max_index];
                all_time_best_fitness = population_fitnesses[max_index];
            }
        }
    }
    
    /**
     * Selects the parents from the current generation, placing them into the
     * parents array.
     * 
     * This implementation uses rank-based selection:
     *                  P(x) = 2/|P| * (|P| - rank(x)) / (|P| + 1)
     */
    private void selectParents() {
        // Map the ranks to the probability distribution shown above
        double[] dist = new double[population_size];
        double p = population_size;
        double sum = 0;
        for(int i = 0; i < population_size; i++) {
            dist[i] = 2/p * (p - i) / (p + 1);
            sum += dist[i];
        }
        // Populate the parent array
        for(int i = 0; i < parents.length; i++) {
            // Pick from the distribution
            double prob = rand.nextDouble();
            int sel = -1;
            do {
                sel++;
                prob -= dist[sel];
                //System.out.println(i + " " + sel +  "   " + prob + "    " + dist[sel]);
            } while(prob > dist[sel]);
            // Clone the vector (to avoid write over the original) and store it
            // in the parents array.
            parents[i] = population[sel].clone();
        }
    }
    
    /**
     * Crosses two vectors using uniform crossover.
     * @param a A vector to be crossed
     * @param b A vector to be crossed
     * @return An array of the 2 resulting crossed vectors.
     */
    private Vector[] cross(Vector a, Vector b) {
        // Clone the vectors
        a = a.clone(); b = b.clone();
        // Iterate through each vector value, testing if the values will be
        // crossed
        for(int i = 0; i < a.getLength(); i++) {
            // Generate a random double to see if crossover will occur
            if(rand.nextDouble() <= crossover_rate) {
                // Swap values
                double temp = a.get(i);
                a.set(i, b.get(i));
                b.set(i, temp);
            } 
        }
        Vector[] crossed = {a, b};
        return crossed;
    }
    
    /**
     * Crosses the parents in the parent array to create children who are placed
     * into the children array.
     */
    private void crossParents() {
        // Iterate through pairs of parents
        for(int i = 0; i < parents.length; i = i + 2) {
            Vector[] crossed = cross(parents[i], parents[i+1]);
            children[i] = crossed[0];
            children[i+1] = crossed[1];
        }
    }
    
    /**
     * Mutates a vector with uniform probability. Utilizes creep because the
     * vector elements are real valued.
     * @param a
     * @return 
     */
    private Vector mutate(Vector a) {
        a = a.clone();
        for(int i = 0; i < a.getLength(); i++) {
            double std_dev = 100;
            // Generate a random double to see if mutation will occur
            if(rand.nextDouble() <= mutation_rate) {
                // Mutate
                double update = a.get(i);
                // Creep takes from a normal distribution centered at 0 with
                // a tuned/determined std deviation
                update += rand.nextGaussian()*std_dev;
                a.set(i, update);
            } 
        }
        return a;
    }
    
    /**
     * Mutates all the children in the children array, modifying them in place.
     */
    private void mutateChildren() {
        // Iterate through children
        for(int i = 0; i < children.length; i++) {
            children[i] = mutate(children[i]);
        }
    }
    
    /**
     * Places the children into the population, replacing certain individuals.
     * 
     * In this implementation, children are placed randomly into the population,
     * one at a time, but only if they are better than that individual.
     */
    private void replaceChildren() {
        for(int i = 0; i < children.length; i++) {
            int select = rand.nextInt(population_size); // Compute random index
            if( computeFitness(children[i]) > population_fitnesses[select]) {
                // Only replace if it is better
                population[select] = children[i];
            }
        }
    }
    
    /**
     * Initialize the population of vectors based on the desired topology of the
     * network.
     */
    private void initializePopulation() {
        // Initialize the arrays that hold our populations
        population = new Vector[population_size];
        parents = new Vector[population_size / NUM_CHILDREN_DIVISOR];
        children = new Vector[population_size / NUM_CHILDREN_DIVISOR];
        population_rank  = new int[population_size];
        population_fitnesses = new double[population_size];
        
        // Fill the initial population
        for(int i = 0; i < population.length; i++) {
            MLP network = new MLP(topology, sim); // Create a network with a 
                                                  // given topology.
            network.randPopWeights(); // Randomly populate the weights in the 
                                      // network.
            population[i] = network.toVec(); // Convert the netowrk to a vector 
                                             // and store it in our array.
        }
    }
    
    /* ---- DISPLAY / DEBUGGING FUNCTIONS ---- */
    private void printPopRank() {
        System.out.print("Current fitness rankings: [ ");
        for(int i = 0; i < population_rank.length; i++) {
            if(i%10 == 0) {System.out.println();}
            System.out.print(population_rank[i] + " ");
        }
        System.out.println("]");
    }
    
    private void printPopFitness() {
        System.out.print("Current fitnesses: [ ");
        for(int i = 0; i < population_fitnesses.length; i++) {
            if(i%10 == 0) {System.out.println();}
            System.out.print(population_fitnesses[i] + " ");
        }
        System.out.println("\n]");
    }
    
    private void printAllTimeBest() {
        System.out.print("ALL TIME BEST FITNESS: ");
        System.out.println(all_time_best_fitness);
    }
}
