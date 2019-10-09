 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

// import libraries
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;

// import exceptions
import java.io.IOException;

/**
 * Class to take in a file containing a dataset and
 * store its values in memory.
 * Specifically, each example in the file is stored 
 * the appropriate Set (also based on the input file)
 * 
 * The input file is assumed to have an agreed upon format
 * so that DataReader can be generalized
 * @author natha
 */
public class DataReader {
    
    // variable to store file name
    private final String file_name;
    // array storing information on the data of the form
    // { num_attr, num_examples, num_classes }
    private final int[] data_summary = new int[4];
    // array storing class names. in our input files, 
    // classes are assigned to a number from 0 up to c, the number of
    // classes. The string can be accessed using this array
    private String[] class_names;
    // variable to store number of subsets
    private int num_subset = 10;
    // array to store the subsets used in 10-fold cross validation
    private Set[] subsets = new Set[num_subset];
    // Set class to store the validation set
    private Set validation_set;
    // array of similarity matrices for computing categorical distances
    private SimilarityMatrix[] sim_matr = null;

    /**
     * Constructor to take input from file file_name
     * @param file_name 
     */
    public DataReader(String file_name){
        // populate global variable file_name
        this.file_name = file_name;
        
        // read and process file
        try{ readFile(); }
        catch(IOException e){
            System.err.println("Opening file error");
            e.printStackTrace();
        }
    }
    
    /**
     * method to read in the data from the file
     * @throws IOException 
     */
    private void readFile() throws IOException {
        // construct file to be read
        //File file = new File("../../../../Preprocessing/DataFiles/" + file_name);
        File file = new File("../Preprocessing/ProcessedDataFiles/" + file_name);
        
        // construct the buffered reader
        BufferedReader br = new BufferedReader(new FileReader(file));
        
        // populate global array data_summary with appropriate values
        String line = br.readLine();
        String[] split_line = line.split(",");
        for (int i = 0; i < 4; i++){ this.data_summary[i] = Integer.parseInt(split_line[i]); }
        
        // declare and instantiate variables for set class
        int num_attr = this.getNumAttributes();
        int num_examples = this.getNumExamples();
        int num_classes = this.getNumClasses();
        int num_cat_attr = this.getNumCatAttributes();
        
        // instantiate sim_matr to correct size
        this.sim_matr = new SimilarityMatrix[num_cat_attr];
        for (int i = 0; i < num_cat_attr; i++){
            try{ this.sim_matr[i] = readMatrix(br); }
            catch(IOException e){ System.err.println("Unexpected end of file"); }
        }
        
        // if num_classes is -1, then data set is regression
        if (num_classes == -1){ this.class_names = null; }
        // otherwise, the data set is classification
        else{
            // initialize class_names array to correct size
            this.class_names = new String[num_classes];

            // populate global array class_names with appropriate values
            line = br.readLine();
            split_line = line.split(",");
            for (int i = 0; i < num_classes; i++){ this.class_names[i] = split_line[i]; }
        }
        
        // initialize each element in the subsets array
        for (int i = 0; i < this.num_subset; i++){ this.subsets[i] = new Set(num_attr, num_classes, this.class_names); }
        
        // iterate through file line-by-line to populate examples array
        for (int i = 0; i < num_examples; i++){
            line = br.readLine();
            Example temp = new Example(line, num_attr);
            this.subsets[temp.getSubsetIndex()].addExample(temp);
        }
        
        // set validation_set as subset[0]
        this.validation_set = this.subsets[0];
        
        // close buffered reader
        br.close();
    }
    
    /**
     * method to instantiate and populate a matrix directly 
     * from the input file
     * 
     * returns the populated matrix
     * 
     * @param br
     * @return
     * @throws IOException 
     */
    private SimilarityMatrix readMatrix(BufferedReader br) throws IOException {
        // read in matrix info
        String line = br.readLine();
        String[] data = line.split(",");
        
        int attr_index = Integer.parseInt(data[0]);
        int num_options = Integer.parseInt(data[1]);
        int num_bins = Integer.parseInt(data[2]);

        // initialize matrix
        SimilarityMatrix mtr = new SimilarityMatrix(attr_index, num_options, num_bins);

        // populate matrix rows
        for (int j = 0; j < num_options; j++){
            line = br.readLine();
            mtr.addRow(j, line);
        }
        
        // return matrix
        return mtr;
    }
    
    // private getter methods for relevant variables
    private int getNumAttributes(){ return this.data_summary[0]; }
    //private int getNumExamples(){ return this.data_summary[1]; }
    private int getNumClasses(){ return this.data_summary[2]; }
    private int getNumCatAttributes(){ return this.data_summary[3]; }
    
    // public getter methods
    public String[] getClassNames(){ return this.class_names; }
    public Set[] getSubsets(){ return this.subsets; }
    public Set getValidationSet(){ return this.validation_set; }
    public SimilarityMatrix[] getSimMatrices(){ return this.sim_matr; }
    public int getNumExamples() { return this.data_summary[1]; };
    
}
