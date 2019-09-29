This repository contains coursework for CSCI 447 (Machine Learning)

File Structure of the Repository

Each directory in the first level of the repository contains the files for an assignment in the machine learning course.
These directories are guaranteed to have some of the same directories:
OrigDataFiles contains the original data files that our algorithms uses
DataDescriptionFiles contains files describing each of the data sets that we are using
Preprocessing contains a .py file that takes in the files within OrigDataFiles and outputs them to a new file with a standard format. These files are then placed in a directory called ProcessedDataFiles inside of Preprocessing
Report contains a written pdf report of our assignment as well as the files needed for compiling the pdf
There is also a directory that is a java project. The java project is the implementation of the actual machine learning algorithm.

We now give a summary of each algorithm

Naive Bayes is a classification algorithm based in Bayes Decision Theory. It is trained on a given data set made up of examples. Each example contains a classification and a list of attributes. We decided to put all attributes with continuous values into bins since our implementation of Naive Bayes must have discrete input.
Given an example x in X, with attributes a1, a2, ..., ad and belonging to class c in C, the algorithm will classify x by choosing the maximum probability P(c|a1, a2, ..., ad) from all c in C
