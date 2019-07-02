# Schrodinger
This program uses a numerical solver for the schrodinger equation to generate a
training data set used in a supervised machine learning program.
Once the trainng data is generated it is used to create a deep neural network with three hidden layers
This neural network is then used to predict the eigenvalue of a given potential without
carrying out any numerical analysis.

When running the program for the first time the assumption is that you will need 
both a training data set and a test data set as well as a new neural network.
These settings will be set to TRUE by default. You only need to generate the data
once. Afterwards it is best to set Generate_Training_Data, Generate_Test_Data
and Create_Neural_Net to FALSE. If the test results are inconsistent then it is best
to increase the sample size and the sample range for the training data set.
It will allow the neural network to be more precise.
