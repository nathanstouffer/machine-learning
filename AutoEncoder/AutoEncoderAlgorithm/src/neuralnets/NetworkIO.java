/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnets;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import neuralnets.layer.IActFunct;
import neuralnets.layer.Layer;
import neuralnets.layer.Linear;
import neuralnets.layer.Logistic;
import neuralnets.layer.Matrix;
import neuralnets.layer.Vector;

/**
 *
 * @author natha
 */
public class NetworkIO {
    
    /**
     * method to write a network to the specified file
     * @param network
     * @param fname 
     */
    protected static void writeLayers(INeuralNet network, String fname) throws FileNotFoundException {
        // create file name
        File fout = new File("../Networks/" + fname);
        
        // construct writer
        PrintWriter writer = new PrintWriter(new FileOutputStream(fout));
        // output string
        String output = Integer.toString(network.getNumLayers()) + "\n";
        
        // iterate through layers
        for (int i = 0; i < network.getNumLayers(); i++) {
            Layer layer = network.getLayer(i);
            output += layer.toString();
        }
        
        // write network
        writer.write(output);
        // close writer
        writer.close();
    }
    
    protected static Layer[] readLayers(String fname) throws FileNotFoundException, IOException {
        // create file
        File fin = new File("../Networks/" + fname);
        // check if file exists
        if (!fin.exists()) { System.err.println("File does not exist."); return null; }
        
        // construct buffered reader
        BufferedReader reader = new BufferedReader(new FileReader(fin));
        
        // read header information
        int num_layers = Integer.parseInt(reader.readLine());
        Layer[] layers = new Layer[num_layers];
        
        // read layers
        for (int i = 0; i < num_layers; i++) {
            // get header infor
            String[] header = reader.readLine().split(",");
            // construct activation function
            IActFunct act_funct;
            if (header[0].equals("logistic")) { act_funct = new Logistic(); }
            else if (header[0].equals("linear")) { act_funct = new Linear(); }
            else { act_funct = null; System.err.println("activation function not available"); }
            // construct empty matrix
            int num_rows = Integer.parseInt(header[1]);
            int num_col = Integer.parseInt(header[2]);
            Matrix weights = new Matrix(num_rows, num_col);
            
            // populate matrix
            for (int j = 0; j < num_rows; j++) {
                String[] line = reader.readLine().split(",");
                // construct empty row
                Vector row = new Vector(num_col);
                // populate row
                for (int k = 0; k < num_col; k++) { 
                    double weight = Double.parseDouble(line[k]);
                    row.set(i, weight); 
                }
                // add row to weight matrix
                weights.setRow(j, row);
            }
            // construct new layer and add to layers array
            layers[i] = new Layer(act_funct, weights);
        }
        
        // close buffered reader
        reader.close();
        
        // return network
        return layers;
    }
    
}
