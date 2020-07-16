import pandas as pd
import numpy as np
import copy

def sort_data(dist_df):
	sorted_dist_df = dist_df.sort_values(by = "Dist")
	return sorted_dist_df
	
def normalize(data,target,mode):
    mean_vector = np.mean(data)
    std_vector = np.std(data)
    normalized_data = pd.DataFrame()
    if mode == "classification":
        for i in data.columns:
            if i != target:
                normalized_data[str(i)+"_norm"] = (data[i]-mean_vector[i])/std_vector[i]
        normalized_data[target] = data[target]
    else:
        for i in data.columns:
            normalized_data[str(i)+"_norm"] = (data[i]-mean_vector[i])/std_vector[i]
    output={"mean":mean_vector,"std":std_vector,"data_norm":normalized_data}
    return output

def distance(train,test,target):
    output={}
    traincols = train.columns
    testcols = test.columns
    disttrain = pd.DataFrame()
    disttest = pd.DataFrame()
    for j in train.columns:
        if j != target:
            disttrain[j]=train[j]
    for j in test.columns:
        if j != target:
            disttest[j]=test[j]
    for i in range(len(test)):
        k = np.sqrt(np.sum((disttrain.values - disttest.loc[test.index[i]].values)**2,axis=1))
        output[i]= copy.deepcopy(k)
    return output

def predictData(train_data,test_data,kvalue,target,mode='regression',cv=False):
    train_norm = normalize(train_data,target,mode)
    train_norm_data = train_norm['data_norm']
    test_norm = normalize(test_data,target,mode)
    test_norm_data = test_norm['data_norm']
    dist={}
    dist = distance(train_norm_data,test_norm_data,target)
    uniq_target = pd.unique(train_data[target])
    tss= 0
    nerr= 0
    predarray=[]
    error=[]
    acarr=[]
    for j in range(len(test_data)):
        td= copy.deepcopy(train_norm_data)
        td["Dist"]=dist[j]
        td1 = sort_data(td)
        idx = td1.index.values
        idx_val = idx[0:kvalue]
        if mode=='regression':
            mean = 0
            for p in idx_val:
                mean+=train_data.loc[p][target]
            mean=mean/kvalue
            nearest = pd.DataFrame(columns = train_data.columns)
            for m in idx_val:
                nearest.loc[m] = train_data.loc[m]
            mpred = mean
            predarray.append(mpred)
            actual = test_data.loc[j][target]
            acarr.append(actual)
            pred = actual - mpred
            cost = pred**2
            error.append(cost)
            nerr+=cost
        elif mode=='classification':
            target_pred_arr=[]
            for p in idx_val:
                target_pred_arr.append(train_data.loc[p][target])
            nearest = pd.DataFrame(columns = train_data.columns)
            for m in idx_val:
                nearest.loc[m] = train_data.loc[m]
            pred_labels = nearest[target]
            count_dict = {}
            count_list=[]
            for i in uniq_target:
                count_dict[i]=0
            for i in pred_labels:
                count_dict[i] += 1
            max_idx= 0
            max_label=""
            for i in pred_labels:
                if count_dict[i] > max_idx:
                    max_idx = count_dict[i]
                    max_label = i
            mpred = max_label
            predarray.append(mpred)
            actual = test_data.loc[j][target]
            acarr.append(actual)
            err = 0
            if actual == mpred:
                err = 0
            else:
                err = 1
                nerr+=1
            error.append(err)
    #print("Predicted values are : ",predarray)
    #print("")
    #print("Actual values are : ",acarr)
    #print("")
    #print("Errors : ",error)
    if cv==True:
        return nerr
    result_dict={}
    if mode=='classification':
    	result_dict["errors"] = error
    	result_dict["predicted"] = predarray
    	result_dict["actual"] = acarr
    	result_dict["accuracy"] = ((1-(sum(error)/len(error)))*100)
    	return result_dict
    if mode=='regression':
    	result_dict["errors"] = error
    	result_dict["predicted"] = predarray
    	result_dict["actual"] = acarr
    	return result_dict
	
