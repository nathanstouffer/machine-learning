##############################################
#Kevin Browder, Nathan Stouffer, Andy Kirby, Eric Kempf
#Program for importing data from Machine Learning example data and exports
#processed data as two csv file. One with original data with sets for 10 fold CV
#and a header that algorithm code need to run Nieve Bayes. A second file scrambles
#10% of the attributes 
#############################################
import pandas as pd
import itertools
import numpy as np
import random
from random import choice      


#data class imports and preprocesses data
#and exports to .csv for use by a Nieve Bayes
class data:
    #class variables 
    df = pd.DataFrame()
    header = ''
    name = ''
    #contructor that reads in datafile as dataframe and formats for importing into Java Nieve Bayes algorithm
    #Prameters: int location of class column, bool if data contains missing data, bool if data is pre binned, string filename
    def __init__(self, classloc, missing, binning, filename):
        global name
        global header
        global df
        name = filename
        #lists to assign sets and record bin #
        sets = []
        bins = []
        #iterators for assigning sets
        a = 0
        i = 0
        
        df = pd.read_csv('..\\DataFiles\\' + filename + '.data', header = None)
        #create list of iterating values from 0-9 distrubeted equally across rows
        while(i < len(df)):
            sets.append(a)
            a += 1
            if(a >= 10):
                a = 0
            i += 1
        #placing class column at 0 loc
        df.rename(columns={classloc:'Class'}, inplace=True)
        cols_to_order = ['Class']
        new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
        df = df[new_columns]
        #randomize all examples
        df = df.sample(frac=1).reset_index(drop=True)
        #rename attributes 0-num of attribues after moving class column to front 
        fixedcolumns = ['Class']
        for i in range(len(df.columns)-1):
            fixedcolumns.append(i)
        df.columns = fixedcolumns
        
        #assign sets to examples   
        df.insert(1, 'Sets', sets)
        #replace missing values with random value
        #replace y with 1 and n with 0
        if missing:
            df = df.replace(to_replace ='?', value = choice(['1', '0']))
            df = df.replace(to_replace ='y', value = '1')
            df = df.replace(to_replace = 'n', value = '0')
        #Quantile-based discretization into five bins
        #also generates list with number of bins for each attribute
        if binning:
            for i in range(len(df.columns)-2):
                df[i] = pd.qcut(df[i], 5, labels=False, duplicates='drop')
                bins.append('5')
        #generate list with number of bins for each attribute when data is pre discritized
        else:
            for i in range(len(df.columns)-2):
                bins.append(str(int(df[i].max()) + 1)) 
        #converte class names to ints for reading by algorithm 
        classes = (df.Class.unique())
        classes = np.sort(classes)
        for i in range(len(classes)):
            df['Class'] = df['Class'].replace(classes[i], i)
        #generate first two formated information lines of output  
        firstRow = str(df['Class'].nunique()) + "," + str(len(df.columns)-2) + "," + str(len(df)) + '\n'     
        header = firstRow + '*,*,' + ','.join(map(str, bins)) + '\n' + ','.join(map(str, classes)) + '\n'
    #export original processed data to .csv file
    def exportOriginal(self):
        #outputting to .csv
        with open('Datafiles\\' + name + '.csv', 'w') as fp:
            fp.write(header)
            fp.write((df.to_csv(index=False, header = False)))
    #export processed data with 10% of the attributes scrambled
    def exportScramble(self):
        global df
        #determine number of attributes to scramble, if less than 10 attributes 1 will be scrambled
        numtoscramble = 0
        if int((len(df.columns)-2)/10) < 1:
            numtoscramble = 1
        else:
            numtoscramble = int((len(df.columns)-2)/10)
        #randomly selecting which attributes to scramble
        columnscramble = [random.randint(0,len(df.columns)-3) for i in range(numtoscramble)]
        #scrambling attributes
        for column in columnscramble:
            df[column] = df[column].sample(frac=1).reset_index(drop=True)
        #outputting to .csv
        with open('Datafiles\\' + name + 'scrambled.csv', 'w') as fp:
            fp.write(header)
            fp.write((df.to_csv(index=False, header = False)))
            
#creating objects for each of the data files and outputting original and scrambled files
glass = data(10, False, True, 'glass')
glass.exportOriginal()
glass.exportScramble()
housevotes = data(0, True, False, 'house-votes-84')
housevotes.exportOriginal()
housevotes.exportScramble()
iris = data(4, False, True, 'iris')
iris.exportOriginal()
iris.exportScramble()
soybean = data(35, False, False, 'soybean-small')
soybean.exportOriginal()
soybean.exportScramble()
wdbc = data(1, False, True, 'wdbc')
wdbc.exportOriginal()
wdbc.exportScramble()

