import pandas as pd
import numpy as np
import copy as cp
import knn_18234 as knn
import matplotlib.pyplot as plt
import naive_bayes_18234 as nb
import DecisionTree_18234 as dt

def apply_crossvalidation(data,model,target,kfold=10,lb=2,ub=10,mode='regression',depth=5,incols=[]):  
    if(model=='knn'):
        c = split_data(data,kvalue=kfold)
        blist=[]
        for k in range(lb,ub+1):
            tvector=[]
            tss = 0 
            for i in range(kfold):
                train_data = c[i]["train"]
                validation_data = c[i]["validation"]
                v = validation_data.to_numpy()
                valid = pd.DataFrame(v,columns=data.columns)
                tss += knn.predictData(train_data,valid,k,target,mode=mode,cv=True)
            blist.append(tss/kfold)
            print("Message : Mean Cross-validation error for k = "+str(k)+" done...");
        plt.title("Cross Validation plot")
        plt.xlabel("K value")
        plt.ylabel("Mean CV error")
        plt.plot(np.arange(lb,ub+1),blist)
        print("The Mean CV error : ",blist)
        plt.show()
    if(model=='naive_bayesian'):
        c = split_data(data,kvalue=kfold)
        for i in range(kfold):
            print("For kfold = ",i+1)
            train_data = c[i]["train"]
            validation_data = c[i]["validation"]
            v = validation_data.to_numpy()
            valid = pd.DataFrame(v,columns=data.columns)
            nb.naive_bayes(train_data,valid,'Code')
    if(model == 'decision_tree'):
        c = split_data(data,kvalue=kfold)
        cv_err=[]
        for d in range(1,depth+1):
            print("For depth = ",d)
            terr= 0
            for i in range(kfold):
                print("For kfold =",i)
                train_data = c[i]["train"]
                validation_data = c[i]["validation"]
                tree_pred = dt.build_tree(train_data,incols,target,d)
                v = validation_data.to_numpy()
                valid = pd.DataFrame(v,columns=data.columns)
                err = dt.decisionTreePrediction(tree_pred,valid,incols,target,mode="cv")
                terr+=err
            cv_err.append(terr)
        plt.plot(np.arange(1,depth+1),cv_err)
        plt.xlabel("Depth")
        plt.ylabel("CV error")
        plt.show()
            
		
def split_data(data,kvalue= 10):
	# k-value will be provided by user or else by default it will be 10
	if kvalue == 0 or kvalue == 1 or kvalue >= data.shape[0]:
		return "Split not possible ... "
	CV_dict={}
	block_size = int(data.shape[0]/kvalue)
	block={}
	for i in range(kvalue):
		block[i] = data.loc[i*block_size : (i+1)*block_size-1]
	block[i+1] = data.loc[(i+1)*block_size : ]
	data_output={}
	for i in range(kvalue+1):
		data_output["validation"] = block[i]
		temp=[]
		for j in range(kvalue+1):
			if i != j :
				temp.append(block[j])
		data_output["train"] = pd.concat(temp)
		CV_dict[i] = cp.deepcopy(data_output)
		CV_dict["kvalue"]=kvalue
	return CV_dict


