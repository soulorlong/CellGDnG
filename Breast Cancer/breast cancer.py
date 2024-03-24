import pandas as pd
import numpy as np
from numpy import nan
nan=0
path  = './case_study/dataset1/'


import warnings
warnings.filterwarnings("ignore")
from itertools import product as product


LRI_gene = pd.read_csv(path + 'LRI_fix.csv',header=None,index_col=None).to_numpy()[:,0:2]

from sklearn.preprocessing import MinMaxScaler
dt = pd.read_csv('./case_study/breast.csv',index_col = 0,header=None)
#a = np.array(dt.loc["Cell"])
dict = {}
dict["cell"] = np.array(dt.loc["cell"])


for i in range(1,dt.shape[0]):
    dict[dt.index[i]] = np.array(dt.loc[dt.index[i]],dtype = float)

path  = './case_study/'
savepath = './Breast_Cancer/fix2/'
def sigmoid(x):
    return 1/(1+np.exp(-(x-6)))
#0=Breasttumer, 1=Myeloid,  2=T,  3=B,  4=Stromal,  5=Immune

Breasttumer_index = np.where(dict['celltype'] == 0)[0]
Myeloid_index = np.where(dict['celltype'] == 1)[0]
T_index = np.where(dict['celltype'] == 2)[0]
B_index = np.where(dict['celltype'] == 3)[0]
Stromal_index = np.where(dict['celltype'] == 4)[0]
Immune_index = np.where(dict['celltype'] == 5)[0]

print(Breasttumer_index)
print(Myeloid_index)

print(T_index)
print(B_index)
print(Stromal_index)
print(Immune_index)


for i in range(6):
    for j in range(6):
        exec('mult_score{}{} = 0'.format(i,j))  #The intercellular communication score based on the filtered LRIs and the expression product approach.

for i in range(6):
    for j in range(6):
        exec('thrd_score{}{} = 0'.format(i,j)) #The intercellular communication score based on the filtered LRIs and the expression thresholding approach.

for i in range(6):
    for j in range(6):
        exec('mult_list{}{} = []'.format(i,j))  #LRIs between two cell types in expression product approach

for i in range(6):
    for j in range(6):
        exec('mult_list_s{}{} = []'.format(i,j))  #LRI score between two cell types in expression product approach

for i in range(6):
    for j in range(6):
        exec('thrd_list{}{} = []'.format(i,j))  #expression thresholding approach

for i in range(6):
    for j in range(6):
        exec('thrd_list_s{}{} = []'.format(i,j))

for i in range(6):                              #specific expression  approach
    for j in range(6):
        exec('spec_score{}{} = 0'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('spec_list{}{} = []'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('spec_list_s{}{} = []'.format(i,j))

for i in range(6):                              #total exp
    for j in range(6):
        exec('total_score{}{} = 0'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('total_list{}{} = []'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('total_list_s{}{} = []'.format(i,j))






g = 0
for i in LRI_gene:
    if i[0] in dict and i[1] in dict:

        g = g + 1
        print('Number of LRIs：',g)


        Breasttumer_l=1/Breasttumer_index.shape[0]*sum(dict[i[0]][Breasttumer_index])  #expression product approach
        Breasttumer_r=1/Breasttumer_index.shape[0]*sum(dict[i[1]][Breasttumer_index])
        Myeloid_l=1/Myeloid_index.shape[0]*sum(dict[i[0]][Myeloid_index])
        Myeloid_r=1/Myeloid_index.shape[0]*sum(dict[i[1]][Myeloid_index])
        T_l=1/T_index.shape[0]*sum(dict[i[0]][T_index])
        T_r=1/T_index.shape[0]*sum(dict[i[1]][T_index])
        B_l=1/B_index.shape[0]*sum(dict[i[0]][B_index])
        B_r=1/B_index.shape[0]*sum(dict[i[1]][B_index])
        Stromal_l=1/Stromal_index.shape[0]*sum(dict[i[0]][Stromal_index])
        Stromal_r=1/Stromal_index.shape[0]*sum(dict[i[1]][Stromal_index])
        Immune_l=1/Immune_index.shape[0]*sum(dict[i[0]][Immune_index])
        Immune_r=1/Immune_index.shape[0]*sum(dict[i[1]][Immune_index])


        l_list = [Breasttumer_l,Myeloid_l,T_l,B_l,Stromal_l,Immune_l]
        r_list = [Breasttumer_r,Myeloid_r,T_r,B_r,Stromal_r,Immune_r]

        a = b = 0
        for item in product(l_list, r_list):
            exec('mult_score{}{} += {}'.format(a,b, (item[0]*item[1])))
            exec('mult_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
            exec('mult_list_s{}{}.append({})'.format(a,b,(item[0]*item[1])))
            b += 1
            if b == 6:
                b = 0
                a += 1

        mean_l_Breasttumer = np.mean(dict[i[0]][Breasttumer_index])   #expression thresholding approach
        mean_l_Myeloid = np.mean(dict[i[0]][Myeloid_index])
        mean_l_T = np.mean(dict[i[0]][T_index])
        mean_l_B = np.mean(dict[i[0]][B_index])
        mean_l_Stromal = np.mean(dict[i[0]][Stromal_index])
        mean_l_Immune = np.mean(dict[i[0]][Immune_index])


        mean_l = np.mean((mean_l_Breasttumer,mean_l_Myeloid,mean_l_T,mean_l_B,mean_l_Stromal,mean_l_Immune))
        std_l = np.std(dict[i[0]][np.concatenate((Breasttumer_index,Myeloid_index,T_index,B_index,Stromal_index,Immune_index))])
        sum_l = np.sum((mean_l_Breasttumer,mean_l_Myeloid,mean_l_T,mean_l_B,mean_l_Stromal,mean_l_Immune))

        mean_r_Breasttumer = np.mean(dict[i[1]][Breasttumer_index])
        mean_r_Myeloid = np.mean(dict[i[1]][Myeloid_index])
        mean_r_T = np.mean(dict[i[1]][T_index])
        mean_r_B = np.mean(dict[i[1]][B_index])
        mean_r_Stromal = np.mean(dict[i[1]][Stromal_index])
        mean_r_Immune = np.mean(dict[i[1]][Immune_index])


        mean_r = np.mean((mean_r_Breasttumer,mean_r_Myeloid,mean_r_T,mean_r_B,mean_r_Stromal,mean_r_Immune))
        std_r = np.std(dict[i[1]][np.concatenate((Breasttumer_index,Myeloid_index,T_index,B_index,Stromal_index,Immune_index))])
        sum_r = np.sum((mean_r_Breasttumer,mean_r_Myeloid,mean_r_T,mean_r_B,mean_r_Stromal,mean_r_Immune))

        Breasttumer_l=int(mean_l_Breasttumer>mean_l+std_l)
        Breasttumer_r=int(mean_r_Breasttumer>mean_r+std_r)
        Myeloid_l=int(mean_l_Myeloid>mean_l+std_l)
        Myeloid_r=int(mean_r_Myeloid>mean_r+std_r)
        T_l=int(mean_l_T>mean_l+std_l)
        T_r=int(mean_r_T>mean_r+std_r)
        B_l=int(mean_l_B>mean_l+std_l)
        B_r=int(mean_r_B>mean_r+std_r)
        Stromal_l=int(mean_l_Stromal>mean_l+std_l)
        Stromal_r=int(mean_r_Stromal>mean_r+std_r)
        Immune_l=int(mean_l_Immune>mean_l+std_l)
        Immune_r=int(mean_r_Immune>mean_r+std_r)


        l_list = [Breasttumer_l,Myeloid_l,T_l,B_l,Stromal_l,Immune_l]
        r_list = [Breasttumer_r,Myeloid_r,T_r,B_r,Stromal_r,Immune_r]

        a = b = 0
        for item in product(l_list, r_list):
            exec('thrd_score{}{} += {}'.format(a,b, int(item[0]&item[1])))
            exec('thrd_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
            exec('thrd_list_s{}{}.append({})'.format(a,b,int(item[0]&item[1])))
            b += 1
            if b == 6:
                b = 0
                a += 1

        sum_l_Breasttumer = np.sum(dict[i[0]][Breasttumer_index])  # expression thresholding approach
        sum_l_Myeloid = np.sum(dict[i[0]][Myeloid_index])
        sum_l_T = np.sum(dict[i[0]][T_index])
        sum_l_B = np.sum(dict[i[0]][B_index])
        sum_l_Stromal = np.sum(dict[i[0]][Stromal_index])
        sum_l_Immune = np.sum(dict[i[0]][Immune_index])


        sum_r_Breasttumer = np.sum(dict[i[1]][Breasttumer_index])
        sum_r_Myeloid = np.sum(dict[i[1]][Myeloid_index])
        sum_r_T = np.sum(dict[i[1]][T_index])
        sum_r_B = np.sum(dict[i[1]][B_index])
        sum_r_Stromal = np.sum(dict[i[1]][Stromal_index])
        sum_r_Immune = np.sum(dict[i[1]][Immune_index])

        Breasttumer_l = sum_l_Breasttumer
        Breasttumer_r = sum_r_Breasttumer
        Myeloid_l = sum_l_Myeloid
        Myeloid_r = sum_r_Myeloid
        T_l = sum_l_T
        T_r = sum_r_T
        B_l = sum_l_B
        B_r = sum_r_B
        Stromal_l = sum_l_Stromal
        Stromal_r = sum_r_Stromal
        Immune_l = sum_l_Immune
        Immune_r = sum_r_Immune
        l_list = [Breasttumer_l, Myeloid_l, T_l, B_l, Stromal_l, Immune_l]
        r_list = [Breasttumer_r, Myeloid_r, T_r, B_r, Stromal_r, Immune_r]

        a = b = 0
        for item in product(l_list, r_list):
            exec('total_score{}{} += {}'.format(a, b, (item[0]*item[1])))
            exec('total_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
            exec('total_list_s{}{}.append({})'.format(a, b, (item[0]*item[1])))
            b += 1
            if b == 6:
                b = 0
                a += 1



        sp_l_Breasttumer = mean_l_Breasttumer / sum_l
        sp_l_Myeloid = mean_l_Myeloid / sum_l
        sp_l_T = mean_l_T / sum_l
        sp_l_B = mean_l_B / sum_l
        sp_l_Stromal = mean_l_Stromal / sum_l
        sp_l_Immune = mean_l_Immune / sum_l

        sp_l_list = [sp_l_Breasttumer, sp_l_Myeloid, sp_l_T, sp_l_B, sp_l_Stromal, sp_l_Immune ]

        sp_r_Breasttumer = mean_r_Breasttumer / sum_r
        sp_r_Myeloid = mean_r_Myeloid / sum_r
        sp_r_T = mean_r_T / sum_r
        sp_r_B = mean_r_B / sum_r
        sp_r_Stromal = mean_r_Stromal / sum_r
        sp_r_Immune = mean_r_Immune / sum_r

        sp_r_list = [sp_r_Breasttumer, sp_r_Myeloid, sp_r_T, sp_r_B, sp_r_Stromal, sp_r_Immune ]

        a = b = 0
        for item in product(sp_l_list, sp_r_list):  # product(A, B) 和 ((x,y) for x in A for y in B)一样
        # print("sigmoid:%f"%sigmoid(item[0]*item[1]))

            exec('spec_score{}{} += {}'.format(a,b,(item[0]*item[1])))
            exec('spec_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
            exec('spec_list_s{}{}.append({})'.format(a,b,(item[0]*item[1])))
            b += 1
            if b == 6:
                b = 0
                a += 1

# for i in range(8):
#     exec('mult_score{}{} = mult_score{}{}/2'.format(i,i,i,i))
#
# for i in range(8):
#     exec('thrd_score{}{} = thrd_score{}{}/2'.format(i,i,i,i))

for i in range(6):
    for j in range(6):
        with open(savepath + "mult_score.txt","a") as f:
            exec('f.write("mult_score{}{} = %f"%(mult_score{}{}))'.format(i,j,i,j))
            f.write('\n')
for i in range(6):
    for j in range(6):
        with open(savepath + "thrd_score.txt","a") as f:
            exec('f.write("thrd_score{}{} = %f"%(thrd_score{}{}))'.format(i,j,i,j))
            f.write('\n')
for i in range(6):
    for j in range(6):
        with open(savepath + "spec_score.txt", "a") as f:
            exec('f.write("spec_score{}{} = %f"%(spec_score{}{}))'.format(i,j,i,j))
            f.write('\n')
for i in range(6):
    for j in range(6):
        with open(savepath + "total_score.txt", "a") as f:
            exec('f.write("total_score{}{} = %f"%(total_score{}{}))'.format(i,j,i,j))
            f.write('\n')


for i in range(6):
    for j in range(6):
        exec('x = pd.DataFrame(mult_list{}{})'.format(i,j))
        exec('y = pd.DataFrame(mult_list_s{}{},columns=list("3"))'.format(i,j))
        x = x.join(y)
        exec('x.to_csv(savepath + "mult_list{}{}.csv", header=None, index=None)'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('x = pd.DataFrame(thrd_list{}{})'.format(i,j))
        exec('y = pd.DataFrame(thrd_list_s{}{},columns=list("3"))'.format(i,j))
        x = x.join(y)
        exec('x.to_csv(savepath + "thrd_list{}{}.csv", header=None, index=None)'.format(i,j))

for i in range(6):
    for j in range(6):
        exec('x = pd.DataFrame(spec_list{}{})'.format(i, j))
        exec('y = pd.DataFrame(spec_list_s{}{},columns=list("3"))'.format(i, j))
        x = x.join(y)
        exec('x.to_csv(savepath + "spec_list{}{}.csv", header=None, index=None)'.format(i, j))

for i in range(6):
    for j in range(6):
        exec('x = pd.DataFrame(total_list{}{})'.format(i, j))
        exec('y = pd.DataFrame(total_list_s{}{},columns=list("3"))'.format(i, j))
        x = x.join(y)
        exec('x.to_csv(savepath + "total_list{}{}.csv", header=None, index=None)'.format(i, j))


