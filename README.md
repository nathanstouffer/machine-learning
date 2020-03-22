This repository contains coursework for CSCI 447 (Machine Learning)

The code was written primarily in Java and all files are our own work. To be specific, this group consisted of Nathan Stouffer, Andrew Kirby, Kevin Browder, and Eric Kempf. The only exception to this is that all data files are from the UCE Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets.php) and any files found in a ''research'' directory are published papers we read and used in our report.

File Structure of the Repository

Each directory in the first level of the repository contains the files for a project in the machine learning course.
These projects tend to have some of the same directories, although not all of them contain each of the following. They are described as follows:

Preprocessing: This directory contains a number of things. First, there are the files contained in OrigDataFiles. These files consist of examples used to train and evaluate a learner's performance. For each data file, there is a corresponding file in the director DataDescriptionFiles that describes properties of the data set. There is also a .py called Preprocessor. This file takes in the data sets in OrigDataFiles and outputs them in a standard format (that we created). This is so that a Learner can assume a standard format for it's input. The standardized files are then output into the directory ProcessedDataFiles.

DesignDocumentation: This directory contains two primary files. The first is a UML Class diagram depicting the overall plan for our project. The diagram is not in excessive depth but gets the general point across. Accompanying the class diagram is a .pdf explaining aspects of the project using words.

Research: This directory contains files that we came across while researching the algorithm. They were stored in this directory for future reference and for use while writing our report.

Report: This directory consists of the necessary .tex files for our paper. There is also a compiled .pdf of our report. The paper was required to be 10 pages or less.

Output: This directory contains files that summarize the performance of the learner on each data set. The output files consist of the raw performance data. If you are interested in a clearer representation, the .pdf in the Report directory contains a much better synopsis.

*Algorithm: This directory contains the java files for the project. We did not produce a .jar and instead ran the project in our IDE.

There were 4 projects (with 1 extra credit project) in this class.

The first two projects were an implementation of Naive Bayes Classification as well as K-Nearest Neighbor (K-NN). We also implemented a number of clustering algorithms for K-Nearest Neighbor. These two projects, along with discussions of SVM and Decision Trees, occupied the first half of the semester.

From there, the focus of the course moved to Neural Networks. This induced the next two projects (found in the Backprop and PopBased directories). As the name implies, Backpropagation was used to train the neural nets in that project. Specifically, there was a Radial Basis Function (that used some of the clustering algorithms from K-NN) and a Multi-Layer Perceptron (MLP). In the PopBased project, the Radial Basis Function was dropped and we were tasked only with training a MLP. We used three distinct training algorithms: the Genetic Algorithm, Differential Evolution, and Particle Swarm Optimization.

The extra credit project also implemented a Neural Network: an Autoencoder. An autoencoder is a net with one hidden layer where the dimension of the input space is equal to the dimension of the output space. There are multiple uses for an autoencoder (ie lossy data compression). The purpose of ours was feature selection, trained via Backpropagation with a sparsity penalty that tended weights to the output layer towards 0. After training a network, we kept only the input and hidden layer. This makes classification and function approximation easier since the input features would be linearly independent.
