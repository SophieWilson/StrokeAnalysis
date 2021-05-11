## StrokeAnalysis
Various files relating to a large data analysis interested in stroke recovery using neurological questionaire data and clinical. 

###### **Preprocessing files **
(These files are not in this repo as they relate to the actual data used which is not publically available)
datapreprocessing - Inital preprocessing and exploration, this was run first. 
datawrangling.py - Some additional code required to subset outlier class (offshoot of preprocessing)
SMOTE.py - The smote augmentation of data

###### **Analysis scripts**
rfvscode - Final random forest model in python, predicting fitter and non-fitter classes using tuned model
strokecgan - The final iteration of cGAN used for data augmentation, currently worse than SMOTE and so not used further.

###### **Additional**
cGAN - A tutorial document detailing a generalised cGAN, not used in project further.
