import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Uncomment if required for 3-D visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns

# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

count = 0 #GlobalVariable

	
def remove_outlier(dataset, principalDf):
	global count
	print("count "+str(count))
	if(count == 0):
		count += 1
		arr = []
		for i in range(len(principalDf['principal_component_1'])):
			# try for different values of pc1 & pc2
			# pc2 is direction of maximum variance
			if (principalDf.iloc[i]['principal_component_1'] > 500):
				arr.append(i)		
		print(arr)
		
		print("Dataset shape before: " + str(dataset.shape))
		#this step needs to be performed iteratively
		dataset.drop(arr, axis = 0, inplace = True) # drop outliers from data
		print("Dataset shape after: " + str(dataset.shape))
			
		dataset = plot_data(dataset)
		return dataset
	
	elif(count == 1):
		count += 1
		arr = []
		for i in range(len(principalDf['principal_component_1'])):
			if ((principalDf.iloc[i]['principal_component_2'] > 15) or (principalDf.iloc[i]['principal_component_2'] < (-7.5)) or 
				(principalDf.iloc[i]['principal_component_1'] > 5)):
				arr.append(i)		
		print(arr)
	
		print("Dataset shape before: " + str(dataset.shape))
		dataset.drop(arr, axis = 0, inplace = True)
		print("Dataset shape after: " + str(dataset.shape))
	
		dataset = plot_data(dataset)
		return dataset
		

def pca(dataset):
	X = dataset.drop(['Name','md5','legitimate'], axis = 1).values
	y = dataset['legitimate'].values

	#Normalize data before applying PCA
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	#reduce to 2D to better visualize data
	pca = PCA(n_components = 3)
	principalComponents = pca.fit_transform(X)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])
	print(principalDf.head())
	print('Explained variance ratio: {}'.format(pca.explained_variance_ratio_))
	
	return principalDf		
		
		
def plot_data(dataset):
	#Perform PCA
	principalDf = pca(dataset)

	#ScatterPlot # plt.subplot(1,3,1) # subplot1
	# similarly create seperate subplots for Benign & Malicious datapoints at each_iter
	plt.figure(figsize = (10, 10))
	plt.xticks(fontsize = 12)
	plt.yticks(fontsize = 14)
	plt.xlabel('Principal Component 1', fontsize=20)
	plt.ylabel('Principal Component 2', fontsize=20)
	plt.title("Principal Component Analysis on Dataset", fontsize=20)
	targets = ['Benign', 'Malignant']
	colors = ['g', 'r']
	for target, color in zip(targets, colors):
		indicesToKeep = dataset['legitimate'].values == target
		#when projected to a 2D space, isn't linearly separable
		#add hue nd depth concept to view high-dimensional data 
		plt.scatter(principalDf.loc[indicesToKeep, 'principal_component_1'], principalDf.loc[indicesToKeep, 'principal_component_2'],
			principalDf.loc[indicesToKeep, 'principal_component_2'], color = color)
			   
	plt.legend(targets, prop ={'size' : 15})
	plt.show()

	# # #---------------------------------
	""" or use this to view data in high-dimensional space during PCA based feature selection """
	# fig = plt.figure(figsize=(8, 6))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(principal_component_1, principal_component_2, principal_component_3, s=50, alpha=0.6, edgecolors='w')
	# ax.set_xlabel('principal_componnet_1')
	# ax.set_ylabel('principal_componnet_2')
	# ax.set_zlabel('principal_componnet_3')
	# # #---------------------------------

	#drop_outliers from the dataset nd again visualize
	if(count == 0):
		dataset = remove_outlier(dataset, principalDf)
		return dataset
	if(count == 1):
		dataset = remove_outlier(dataset, principalDf)
		return dataset
	else:
		return dataset

		
def extract_data():
	""" Randomize data inplace here for better distribution (rows) or use this in a seperate run"""
	dataset = pd.read_csv('updated_data.csv', low_memory = False)
	print(dataset.head())
	dataset = dataset.sample(frac = 1, random_state = 42).reset_index(drop = True)
	print(dataset.head())
	dataset['legitimate'].replace(1, 'Benign', inplace=True)
	dataset['legitimate'].replace(0, 'Malignant', inplace=True)
	
	#Plot data
	dataset = plot_data(dataset)
	
	return dataset
	
	
if __name__ == '__main__':
	dataset = extract_data()
	print('Final Dataset ' + str(dataset.shape)) #dataset size less than the original one
	
	#save to dataset.pkl for future use
	open('dataset.pkl', 'wb').write(pickle.dumps(dataset))