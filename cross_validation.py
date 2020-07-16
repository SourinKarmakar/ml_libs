import pandas as pd
import numpy as np
import copy as cp
import knn_18234 as knn
import matplotlib.pyplot as plt
import naive_bayes_18234 as nb
import DecisionTree_18234 as dt

def apply_crossvalidation(data,model,target,kfold=10,lb=2,ub=10,mode='regression',depth=5,incols=[]):  
    if model=='knn':
        uniq_target=pd.unique(data[target])
        norm = knn.normalize(data,target,mode)
        norm_data = norm['data_norm']
        c = split_data(norm_data,kvalue=kfold)
        predarray=[]
        blist=[]
        #for all values between the upper and lower bound find the mean CV error
        for k in range(lb,ub+1):
            tvector=[]
            #for each fold
            for i in range(kfold):
                #getting training and test data
                train_data = c[i]["train"]
                validation_data = c[i]["validation"]
                #applying the distance to obtain all the values of distance for each test point
                dist = knn.distance(train_data,validation_data,target)
                #total sum of square
                tss= 0
                if len(validation_data) != 0:
                    #for all training points
                    for j in range(len(validation_data)):
                        #print("\n"+str(i)+" "+str(j))
                        #it won't affect the training set
                        td= cp.deepcopy(train_data)
                        td["Dist"]=dist[j]
                        #sorting according to the distance
                        #td.sort_values(by="Dist")
                        td1 = knn.sort_data(td)
                        #getting the sorted index
                        idx = td1.index.values
                        #take only first k such index
                        idx_val = idx[0:k]
                        if mode=='regression':
                            mean = 0
                            for p in idx_val:
                                mean+=data.loc[p][target]
                            mean=mean/k
                            nearest = pd.DataFrame(columns = data.columns)
                            for m in idx_val:
                                nearest.loc[m] = data.loc[m]
                            mpred = mean
                            block_size = len(data)//kfold
                            actual = data.loc[(i*block_size)+j][target]
                            pred = actual - mpred
                            cost = pred**2
                            tss = tss+cost
                        elif mode=='classification':
                            target_pred_arr=[]
                            for p in idx_val:
                                target_pred_arr.append(data.loc[p][target])
                            nearest = pd.DataFrame(columns = train_data.columns)
                            for m in idx_val:
                                nearest.loc[m] = train_data.loc[m]
                            pred_labels = nearest[target]
                            count_dict = {}
                            count_list=[]
                            for q in uniq_target:
                                count_dict[q]=0
                            for q in pred_labels:
                                count_dict[q] += 1
                            max_idx= 0
                            max_label=""
                            for q in pred_labels:
                                if count_dict[q] > max_idx:
                                    max_idx = count_dict[q]
                                    max_label = q
                            mpred = max_label
                            block_size = len(data)//kfold
                            actual = validation_data.loc[(i*block_size)+j][target]
                            err = 0
                            if actual == mpred:
                                err = 0
                            else:
                                err = 1
                            tss = tss+err
                        tvector.append(tss)
                #print("	         Computation done for fold : ",str(i+1));
            blist.append(sum(tvector)/kfold)
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


