package naivebayesalgorithm;

/**
 *
 * @author natha
 */
public class Driver {

    public static void main(String[] args) {
        DataReader reader = new DataReader("glass.csv");
        NaiveBayes nb = new NaiveBayes();
        
        Set[] sets = reader.getSubsets();
        nb.train(sets[0]);
        
        nb.test(sets[1]);
    }
    
}
