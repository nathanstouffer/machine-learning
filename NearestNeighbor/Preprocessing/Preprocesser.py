##############################################
# Kevin Browder, Nathan Stouffer, Andy Kirby, Eric Kempf
# Program for importing data from Machine Learning example data and exports
# processed data as two csv file. One with original data with sets for 10 fold CV
# and a header that algorithm code need to run Naive Bayes. A second file scrambles
# 10% of the attributes
#############################################
import csv
import numpy as np
import pandas as pd
import random


# data class imports and preprocesses data
# and exports to .csv for use by a Naive Bayes
class data:
    # class variables

    # contructor that reads in datafile as dataframe and formats for importing into Java Nieve Bayes algorithm
    # Prameters: int location of class column, bool if data contains missing data, bool if data is pre binned,
    # string filename
    def __init__(self, classloc, remove, missing, classification, filename, seperator):
        df = pd.DataFrame()
        name = filename
        # lists to assign sets and record bin #
        sets = []
        bins = []
        # iterators for assigning sets
        a = 0
        i = 0
        df = pd.read_csv('..\\OrigDataFiles\\' + filename + '.data', skiprows=remove, sep=seperator, header=None)
        # create list of iterating values from 0-9 distrubeted equally across rows
        while i < len(df):
            sets.append(a)
            a += 1
            if a >= 10:
                a = 0
            i += 1
        # placing class column at 0 loc
        df.rename(columns={classloc: 'Class'}, inplace=True)
        cols_to_order = ['Class']
        new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
        df = df[new_columns]
        # randomize all examples
        df = df.sample(frac=1).reset_index(drop=True)
        # rename attributes 0-num of attribues after moving class column to front
        fixedcolumns = ['Class']
        for i in range(len(df.columns) - 1):
            fixedcolumns.append(i)
        df.columns = fixedcolumns
        # converted class names to ints for reading by algorithm
        if classification:
            classes = (df.Class.unique())
            classes = np.sort(classes)
            for i in range(len(classes)):
                df['Class'] = df['Class'].replace(classes[i], i)
        else:
            backupclass = df['Class']
            df['Class'] = pd.cut(df[i], 10, labels=False)
        # assign sets to examples
        df.insert(1, 'Sets', sets)
        # converts all columns that are strings to integers starting at 0 and indexing by 1
        categorical = []
        catfeat = 0
        for i in range(len(df.columns) - 2):
            if df[i].dtype == object:
                original = df[i].unique()
                replacedict = dict(zip(original, range(len(original))))
                df[i] = df[i].map(replacedict)
                categorical.append("0")
                catfeat += 1
            else:
                categorical.append("1")
        matrices = ''
        print(np.sort(df.Class.unique()))
        for i in range(len(categorical)):
            if categorical[i] == '0':
                matrices += str(i) + ',' + str(len(df[i].unique())) + ',' + str(len(df.Class.unique())) + '\n'
                for a in df[i].unique():
                    for b in np.sort(df.Class.unique()):
                        matrices += str(len(df[(df['Class'] == b) & (df[i] == a)]) / len(df[df['Class'] == b])) + ','
                    matrices += '\n'
        # generate first two rows of formatted output
        if classification:
            header = str(len(df.columns) - 2) + ',' + str(len(df)) + ',' + str(
                df['Class'].nunique()) + ',' + str(catfeat) + '\n' + ',,' + ','.join(
                categorical) + '\n' + matrices + ','.join(map(str, classes)) + ',' + '\n'
        else:
            df['Class'] = backupclass
            header = str(len(df.columns) - 1) + ',' + str(len(df)) + ',' + '-1' + ',' + '\n' + ',,' + ','.join(
                categorical) + '\n' + matrices
        with open('ProcessedDataFiles\\' + name + '.csv', 'w') as fp:
            fp.write(header)
            fp.write((df.to_csv(index=False, header=False)))
            fp.close


# creating objects for each of the data files and outputting original and scrambled files
files = [data(8, 0, False, True, 'abalone', ','), data(6, 0, False, True, 'car', ','),
         data(0, 5, False, True, 'segmentation', ','), data(12, 1, False, False, 'forestfires', ','),
         data(8, 0, False, False, 'machine', ','), data(11, 1, False, False, 'winequality-red', ';'),
         data(11, 1, False, False, 'winequality-white', ';')]
