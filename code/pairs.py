import numpy as np
import pickle
import pandas as pd


interMatrix = pd.read_csv("./data1/ligand-receptor interaction.csv",header=0,index_col=0)
interMatrix = interMatrix.values
rows, cols = interMatrix.shape
print('matrix shape:', interMatrix.shape)
rd_pairs = []
for i in range(rows):
    for j in range( cols):
        rd_pairs.append([i,j,interMatrix[i,j]])

rd_pairs = np.array(rd_pairs).reshape(-1,3)
print(rd_pairs)
np.savetxt("./data1/pair.txt",rd_pairs,fmt='%d')