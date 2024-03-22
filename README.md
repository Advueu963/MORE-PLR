# MORE-PLR
This is the official Reporsitory of the Paper: "MORE–PLR: Multi-Output Regression Employed for Partial Label Ranking"
# Navigation
Files with LR / PLR are the evaluation scripts for the corresponding problem statement. \
__SimpleStart.py__ is the first go to file for a simple example of how to use the models. \
__dataLinks.py__ contains the openML ids of the benchmark data. \
__latex-...__ files are auxiliary files for writing the paper. \
__MORE_models__ contains all models proposed in the Paper and some more. \
__mpp3Script.cmd__ is the SLURM Job processing script that was used to run the experiments. 
The 96 GB mentioned in the paper was automatically controlled through the HPC. \
__data/__ contains all the files for the results. All results are saved to this file \
__scikit-lr/__ contains the pairwise-reduction methods from "Pairwise learning for the partial label ranking problem"
To install run "pip install --verbose --no-build-isolation --editable .; python setup.py build_ext --inplace" in the scikit-lr folder. 