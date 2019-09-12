/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayesalgorithm;

// import libraries
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;

// import exceptions
import java.io.IOException;

/**
 *
 * @author natha
 */
public class DataReader {
    
    // variable to store file name
    private final String file_name;
    // array storing information on the data of the form
    // { num_classes, num_attr, num_examples }
    private final int[] data_summary = new int[3];
    // array storing the number of bins for a corresponding attribute
    // num_bins[0] corresponds to the number of bins for the 0th attribute of a class
    private int[] num_bins;
    // array storing class names. in our input files, 
    // classes are assigned to a number from 0 up to c,
    // the number of classes. The string can be
    // accessed using this array
    private String[] class_names;
    // variable to store number of subsets
    private int num_subset = 10;
    // array to store the subsets
    private Set[] subsets = new Set[num_subset];

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
    
    private void readFile() throws IOException {
        // construct file to be read
        //File file = new File("../../../../Preprocessing/DataFiles/" + file_name);
        File file = new File("../Preprocessing/DataFiles/" + file_name);
        
        // construct the buffered reader
        BufferedReader br = new BufferedReader(new FileReader(file));
        
        // populate global array data_summary with appropriate values
        String line = br.readLine();
        String[] split_line = line.split(",");
        for (int i = 0; i < 3; i++){ this.data_summary[i] = Integer.parseInt(split_line[i]); }
        
        // declare and instantiate variables for set class
        int num_classes = getNumClasses();
        int num_attr = getNumAttributes();
        int num_examples = getNumExamples();
        
        // initialize num_bins array to correct size
        this.num_bins = new int[num_attr];
        
        // populate global array num_bins with appropriate values
        line = br.readLine();
        split_line = line.split(",");
        for (int i = 0; i < num_attr; i++){ this.num_bins[i] = Integer.parseInt(split_line[i + 2]); }

        for (int i = 0; i < num_attr; i++){ this.num_bins[i] = Integer.parseInt(split_line[i + 2]); }
        
        // initialize class_names array to correct size
        this.class_names = new String[num_classes];
        
        // populate global array class_names with appropriate values
        line = br.readLine();
        split_line = line.split(",");
        for (int i = 0; i < num_classes; i++){ this.class_names[i] = split_line[i]; }
        
        // initialize each value in the subsets array
        for (int i = 0; i < this.num_subset; i ++){ this.subsets[i] = new Set(num_classes, num_attr, this.num_bins, this.class_names); }
        
        // iterate through file line-by-line to populate examples array
        for (int i = 0; i < num_examples; i++){
            line = br.readLine();
            Example temp = new Example(line, num_attr);
            this.subsets[temp.getSubsetIndex()].addExample(temp);
        }
        
        br.close();
    }
    
    // getter methods for relevant variables
    public int getNumClasses(){ return this.data_summary[0]; }
    public int getNumAttributes(){ return this.data_summary[1]; }
    public int getNumExamples(){ return this.data_summary[2]; }
    public int[] getNumBin(){ return this.num_bins; }
    public String[] getClassNames(){ return this.class_names; }
    public Set[] getSubsets(){ return this.subsets; }
    
}
