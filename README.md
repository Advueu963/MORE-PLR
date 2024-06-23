# MORE-PLR
This repository contains the source code for the paper **"MORE–PLR: Multi-Output Regression
Employed for Partial Label Ranking"**.
The data shown in the paper can be found in the **data/** folder. \
If you are interested of applying these models yourself, take a look at the *SimpleStart.py* file.
But generally the models were implemented in the typical *sklearn* style, with fit and predict.

# Reproducibility
Firstly make sure to have scikit-lr installed (the current state-of-the-art).
To install execute the following command in the *scikit-lr* folder: `pip install --verbose --no-build-isolation --editable .; python setup.py build_ext --inplace`
General python requirements can be installed via *enviroment.yml*. \
At the end one might need to execute `python setup.py build_ext --inplace` in the root folder, to build the Cython *_overlap_intervals.pyx* file. \
The scripts used for the experiments can be found in *MORE/experiments* \

# Results
All results can be found in `data`.  
We hope the structure of the folders helps navigation.
The folder `results` contains all the results for the different models.
The folder `tables` contains the results represented as tables.
the folder `reports` contains `.html` giving a overview of the results.

# Cite
TBA

# Contact
For indepth questions on MORE feel free to contact S.Thies@campus.lmu.de or JuanCarlos.Alfaro@uclm.es.