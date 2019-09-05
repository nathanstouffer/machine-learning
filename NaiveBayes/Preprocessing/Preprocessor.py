##############################################
#Kevin Browder, Nathan Stouffer, Andy Kirby, Eric Kempf
#Program for importing data from CSV
#
#
#############################################
import pandas as pd
import numpy as np

class data:

    def __init__(self, classloc, missing, attributes, filename):
        new = pd.read_csv("..\\DataFiles\\" + filename + ".data", header = None)
        new.rename(columns={classloc:'class'}, inplace=True)
        cols_to_order = ['class']
        new_columns = cols_to_order + (new.columns.drop(cols_to_order).tolist())
        new = new[new_columns]
        print(new.head())

        
glass = data(10, False, 10, "glass")
housevotes = data(0, False, 16, "house-votes-84")
iris = data(4, False, 4, "iris")
soybean = data(35, False, 35, "soybean-small")
wdbc = data(1, True, 31, "wdbc")
        
