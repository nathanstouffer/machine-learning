/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighboralgorithm;

/**
 * Interface that requires a realizing class to implement a
 * method called reduce. This method will take in a Set object
 * and use an algorithm (specified by the realizing class) to
 * reduce the size of Set argument
 * 
 * @author natha
 */
public interface IDataReducer {
    
    public Set reduce(Set orig);
    
}
