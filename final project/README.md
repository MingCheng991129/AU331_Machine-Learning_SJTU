This folder contains the final project of AU331.

In this project, we design a system *FCCM(Feature extraction-Condition analysis-Condition development Model)* to do the classification of the types of lung cancer(large cell, squamous cell carcinoma, adenocarcinoma, Non-small cell, NA) as well as the stage(I, II, III). We use CT image as input. Also, we predict the survival time of patients by extracting the features of input data in this model. 

Therefore, in FCCM:

- F stands for *feature extraction*, which means we will extract several features from input data using convolutional layer in neural network. 
- C stands for *condition analysis*, which means we will do the classification process (types & stage).
- C stands for *condition development*, which means we will predict the survival time of patients.
