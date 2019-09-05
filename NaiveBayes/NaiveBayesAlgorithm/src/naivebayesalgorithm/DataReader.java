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
    // { class_count, attribute_count, set_size, example_count }
    private final int[] data_summary = new int[4];
    // array to store the examples
    private Example[] examples;

    /**
     * 
     * @param file_name 
     */
    DataReader(String file_name){
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
        File file = new File("../../../../Preprocessing/DataFiles/" + file_name);
        
        // construct the buffered reader
        BufferedReader br = new BufferedReader(new FileReader(file));
        
        // populate global array data_summary with appropriate values
        String line = br.readLine();
        String[] split_line = line.split(",");
        for (int i = 0; i < 4; i++){ data_summary[i] = Integer.parseInt(split_line[i]); }
        
        // TODO: write code to count number of attribute options
        
        // initialize examples array to correct size
        int example_count = data_summary[3];
        examples = new Example[example_count];
        
        // iterate through file line-by-line to populate examples array
        for (int i = 0; i < example_count; i++){
            line = br.readLine();
            examples[i] = new Example(line);
        }
    }
    
    // getter methods for relevant values
    public int getClassCount(){ return data_summary[0]; }
    public int getAttributeCount(){ return data_summary[1]; }
    public int getSetSize(){ return data_summary[2]; }
    public Example[] getExamples(){ return examples; }
    
}
