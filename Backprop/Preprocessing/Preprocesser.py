##############################################
# Kevin Browder, Nathan Stouffer, Andy Kirby, Eric Kempf
# Program for importing data from Machine Learning example data and exports
# processed data as two csv file. One with original data with sets for 10 fold CV
# and a header that algorithm code need to run Nearest Neighbor.
#############################################
import csv
import numpy as np
import pandas as pd
import random


# data class imports and preprocesses data
# and exports to .csv for use by a KNN algorthim
class data:
    # class variables

    # constructor that reads in datafile as dataframe and formats for importing into Java Nearest Neighbor algorithm
    def __init__(self, classloc, remove, classification, filename, separator):
        self.classloc = classloc
        self.remove = remove
        self.classification = classification
        self.filename = filename
        self.separator = separator
        self.numcategoricalvar = 0
        self.header = ''
        self.sets = []
        self.matrices = ''
        self.df = pd.DataFrame()
        self.csvin()
        self.calccategorical()
        self.output()

    def csvin(self):
        # iterators for assigning sets
        a = 0
        i = 0
        self.df = pd.read_csv('OrigDataFiles\\' + self.filename + '.data', skiprows=self.remove, sep=self.separator,
                              header=None)
        # create list of iterating values from 0-9 distributed equally across rows
        while i < len(self.df):
            self.sets.append(a)
            a += 1
            if a >= 10:
                a = 0
            i += 1
        # placing class column at 0 loc
        self.df.rename(columns={self.classloc: 'Class'}, inplace=True)
        cols_to_order = ['Class']
        new_columns = cols_to_order + (self.df.columns.drop(cols_to_order).tolist())
        self.df = self.df[new_columns]
        # randomize all examples
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        # rename attributes 0-num of attributes after moving class column to front

    def calccategorical(self):
        fixedcolumns = ['Class']
        for i in range(len(self.df.columns) - 1):
            fixedcolumns.append(i)
        self.df.columns = fixedcolumns
        # converted class names to ints for reading by algorithm
        if self.classification:
            classes = (self.df.Class.unique())
            classes = np.sort(classes)
            for i in range(len(classes)):
                self.df['Class'] = self.df['Class'].replace(classes[i], i)
        # bins regression values
        else:
            backupclass = self.df['Class']
            self.df['Class'] = pd.cut(self.df['Class'], 10, labels=False)
        # assign sets to examples
        self.df.insert(1, 'Sets', self.sets)
        # converts all categorical variables to integers starting at 0 and indexing by 1
        # also keeps track of which attributes are categorical and which are numerical
        categorical = []
        for i in range(len(self.df.columns) - 2):
            if self.df[i].dtype == object:
                original = self.df[i].unique()
                replace = dict(zip(original, range(len(original))))
                self.df[i] = self.df[i].map(replace)
                categorical.append("0")
                self.numcategoricalvar += 1
            else:
                categorical.append("1")

        for i in range(len(categorical)):
            # calculate matrices for distance metric for all categorical variable
            if categorical[i] == '0':
                self.matrices += str(i) + ',' + str(len(self.df[i].unique())) + ',' + str(
                    len(self.df.Class.unique())) + '\n'
                for a in self.df[i].unique():
                    for b in np.sort(self.df.Class.unique()):
                        self.matrices += str(len(self.df[(self.df['Class'] == b) & (self.df[i] == a)]) / len(
                            self.df[self.df['Class'] == b])) + ','
                    self.matrices += '\n'
            # normalize all numerical variables
            else:
                if self.df[i].max() == self.df[i].min():
                    self.df[i].values[:] = 0
                else:
                    self.df[i] = self.df[i].apply(
                        lambda x: (x - self.df[i].min()) / (self.df[i].max() - self.df[i].min()))
        # generate first two rows of formatted output
        if self.classification:
            self.header = str(len(self.df.columns) - 2) + ',' + str(len(self.df)) + ',' + str(
                self.df['Class'].nunique()) + ',' + str(self.numcategoricalvar) + '\n' + self.matrices + ','.join(
                map(str, classes)) + ',' + '\n'
        else:
            self.df['Class'] = backupclass
            self.header = str(len(self.df.columns) - 2) + ',' + str(len(self.df)) + ',' + '-1' + ',' + str(
                self.numcategoricalvar) + '\n' + self.matrices

    def output(self):
        with open('ProcessedDataFiles\\' + self.filename + '.csv', 'w') as fp:
            fp.write(self.header)
            fp.write((self.df.to_csv(index=False, header=False)))
            fp.close

# creating objects for each of the data files and outputting original and scrambled files
files = [data(8, 0, True, 'abalone', ','), data(6, 0, True, 'car', ','),
         data(0, 5, True, 'segmentation', ','), data(12, 1, False, 'forestfires', ','),
         data(8, 0, False, 'machine', ','), data(11, 1, False, 'winequality-red', ';'),
         data(11, 1, False, 'winequality-white', ';')]
