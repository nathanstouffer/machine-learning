package naivebayesalgorithm;

import java.text.DecimalFormat;

/**
 *
 * @author natha
 */
public class Driver {

    public static void main(String[] args) {
        
        String[] datafiles = {"glass.csv", "iris.csv", "house-votes-84.csv", 
            "soybean-small.csv", "wdbc.csv"}; //skipped "house-votes-84.csv"
        
        for(int f = 0; f < datafiles.length; f++) {
            System.out.println("--- Handling " + datafiles[f] + " data set ---");
            DataReader reader = new DataReader(datafiles[f]);
            NaiveBayes nb = new NaiveBayes();  
            for(int i = 0; i < 10; i++) { // Perform 10-fold cross validation
                System.out.println("Test " + (i+1));

                Set training_set = new Set(reader.getSubsets(), i); // Combine 9 of the subsets

                nb.train(training_set); // Train
                Set testing_set = reader.getSubsets()[i]; // Test with the remaining subset

                int[] predictions = nb.test(testing_set); // Test

                ConfusionMatrix m = new ConfusionMatrix(testing_set, predictions);
                System.out.println("The accuracy was: " + 
                        new DecimalFormat("###.##").format(m.getAccuracy()*100)
                        + "%");

                System.out.println();
            }
            System.out.println("----------------------------------------------");
        }
    }
    
}
