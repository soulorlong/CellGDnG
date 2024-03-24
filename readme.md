# CellGDnG

## Overview

 [overview.pdf](overview.pdf) 

## Environment

- Python==3.6.13

### packages:

- dgl-cu101==0.4.3.post1
- mxnet-cu110==1.9.1
- numpy ==1.19.2
- pandas==1.1.5
- scikit-learn==0.20.3
- scipy==1.5.4
- torch==1.7.0

## Usage

1. We obtain ligand and receptor feature at  [BioTriangle](http://biotriangle.scbdd.com/)

2. Run the model to obtain the LRI, or the user-specified LRI database		

   ```
   python code/data_1.py
   ```

3. Using scoring method (including Specific expression, Expression product and Total expression), the cell-cell communication matrix was finally obtained.		

   ```
   python Breast Cancer/breast cancer.py
   ```

