import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

def divide_data(df,target,uniq):
	divide_dict={}
	for i in uniq:
		temp = df[df[target]==i]
		divide_dict[i] = temp
	return(divide_dict)
	

def getMeanVar(divide_dict,target,uniq):
	msdict={}
	for j in uniq:
		mlist=[]
		slist=[]
		for i in divide_dict[j].columns:
			if i != target:
				mlist.append(np.mean(divide_dict[j][i]))
				slist.append(np.std(divide_dict[j][i]))
		msdict[str(j)+"_"+"meanlist"] = mlist
		msdict[str(j)+"_"+"stdlist"]=slist
	return(msdict)
	

def createCovMatrix(msdict,uniq):
	cov_dict={}
	for i in uniq:
		cov_dict[str(i)+"_varmat"] = np.diag(np.array(msdict[str(i)+"_stdlist"]))
	return(cov_dict)
		

def create_mvnormal(msdict,cov_dict,uniq):
	mvnormaldict ={}
	for i in uniq:
		mvnormaldict[str(i)+"_mv"] = multivariate_normal(mean = msdict[str(i)+"_meanlist"], cov = cov_dict[str(i)+"_varmat"])
	return(mvnormaldict)
	
def calculate_weightage(df,divide_dict,uniq):
	prob_dict={}
	for j in uniq:
		prob_dict[str(j)+'_weightage'] = divide_dict[j].shape[0]/df.shape[0]
	return(prob_dict)
	
def naive_bayes(train,test,target):
	uniq = pd.unique(train[target])
	#print(uniq)
	divide_dict = divide_data(train,target,uniq)
	#print(divide_dict)
	msdict = getMeanVar(divide_dict,target,uniq)
	#print(msdict)
	cov_dict = createCovMatrix(msdict,uniq)
	#print(cov_dict)
	mvnormaldict = create_mvnormal(msdict,cov_dict,uniq)
	#print(mvnormaldict)
	prob_dict = calculate_weightage(train,divide_dict,uniq)
	#print(prob_dict)
	pred_array=[]
	error=[]
	actual=[]
	for i in range(len(test)):
		tpoint=[]
		for k in test.columns:
			if k!=target:
				tpoint.append(test.loc[i][k])
		#print(tpoint)
		lklist=[]
		for j in uniq:
			lklist.append(mvnormaldict[str(j)+"_mv"].pdf(tpoint)*prob_dict[str(j)+"_weightage"])
		index = lklist.index(max(lklist))
		#print(lklist,index)
		pred_array.append(uniq[index])
		actual.append(test.loc[i][target])
		if test.loc[i][target] != uniq[index]:
			error.append(1)
		else:
			error.append(0)
	print("Actual    : ",actual)
	print("Predicted : ",pred_array)
	print("Error     : ",error)
	accuracy = (1-(sum(error)/len(error)))*100
	print("Accuracy : ",accuracy,"%")

	
	
	
