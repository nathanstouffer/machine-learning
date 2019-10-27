/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package networklayer;

/**
 * class to store a matrix. A Matrix is composed of Vectors
 * 
 * Matrices can be added together and multiplied with a vector
 * 
 * @author natha
 */
public class Matrix {
    
    // variable to store a matrix
    // it is composed of an array of vectors
    private Vector[] mtx;
    // variable to store the number of columns
    private int num_col;
    
    /**
     * constructor that initializes all values in the matrix as 0.0
     * @param num_rows
     * @param num_col 
     */
    public Matrix(int num_rows, int num_col){
        // instantiate global variables
        this.mtx = new Vector[num_rows];
        this.num_col = num_col;
        
        // set all values to 0.0
        for (int i = 0; i < num_rows; i++) { this.mtx[i] = new Vector(num_col); }
    }
    
    /**
     * method to add two matrices together and store the
     * result in the matrix the method is called from
     * 
     * An error message is printed if the operation is not valid
     * 
     * @param to_add 
     */
    public void plusEquals(Matrix to_add) {
        // check that adding is valid
        // assume the operation is valid until proven false
        boolean valid = true;
        if (this.getNumRows() != to_add.getNumRows()) { valid = false; }
        if (this.getNumColumns() != to_add.getNumColumns()) { valid = false; }
        
        // if not valid, print an error message
        if (!valid) { System.err.println("Matrix addition not valid"); }
        // otherwise, add the matrices
        else {
            // iterate through rows
            for (int m = 0; m < this.getNumRows(); m++) {
                // add rows together
                this.getRow(m).plusEquals(to_add.getRow(m));
            }
        }
    }
    
    /**
     * method to divide each value in the Matrix by the divisor
     * @param divisor 
     */
    public void divEquals(double divisor) {
        // iterate through Matrix
        for (int i = 0; i < this.getNumRows(); i++) { this.getRow(i).divEquals(divisor); }
    }
    
    /**
     * method to randomly populate the matrix with values
     * between lower and upper
     * 
     * @param lower
     * @param upper 
     */
    protected void randPopulate(double lower, double upper) {
        // call randPopulate on each Vector in the Matrix
        for (int i = 0; i < this.getNumRows(); i++) { this.getRow(i).randPopulate(lower, upper); }
    }
    
    /**
     * method to multiply a matrix and a vector together and
     * return the result in a vector
     * 
     * if the operation is not valid, null is returned
     * 
     * @param vec
     * @return 
     */
    protected Vector mult(Vector vec) {
        // check that multiplying is valid
        if (vec.getLength() != this.getNumColumns()) {
            System.err.println("Matrix-vector multiplication not valid");
            return null;
        }
        // otherwise, apply the operation
        else{
            // instantiate result of product to correct size
            Vector product = new Vector(this.getNumRows());
            
            // iterate through rows, computing the dot product
            for (int m = 0; m < this.getNumRows(); m++) {
                double dot_prod = this.getRow(m).dotProd(vec);
                product.set(m, dot_prod);
            }
            
            // return the product
            return product;
        }
    }
    
    protected int getNumColumns() { return this.num_col; }
    protected int getNumRows() { return this.mtx.length; }
    protected Vector getRow(int index) { return this.mtx[index]; }

}
