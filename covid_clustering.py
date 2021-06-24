"""
covid_clustering.py
"""
import math 
import statistics 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# load dataset final.csv
data = pd.read_csv("final.csv")

# initialize and copy data to attribute variable lists 

# A. country names 
countries = data.iloc[:,0].copy()
# B. total cases / 1M pop
cases = data.iloc[:,1].copy()
# C. total deaths / 1M pop
deaths = data.iloc[:,2].copy()
# D. total tests / 1M pop
tests = data.iloc[:,3].copy()
# E. hospital beds / 1K pop
beds = data.iloc[:,4].copy()
# F. UHC service coverage index
UHC = data.iloc[:,5].copy()
# G. smoking prevalence (% of adult population)
smoking = data.iloc[:,6].copy()

# convert data into numeric types
cols = ["Total Cases / 1M pop", "Deaths / 1M pop", "Tests / 1M pop", \
        "Hospital Beds / 1K pop", "UHC Service Coverage Index", \
        "Smoking Prevalence (% of total adult population)"]
for col in cols:  # iterate over chosen columns
	data[col] = pd.to_numeric(data[col])

'''
# compute and output summary statistics of each attribute
attributes = [cases, deaths, tests, beds, UHC, smoking]
for i in attributes:
    print("Mean: ", np.mean(i))
    print("Min: ", np.min(i))
    print("Min Country: ", countries[np.argmin(i)])
    print("Max: ", np.max(i))
    print("Max Country: ", countries[np.argmax(i)])
    print("Median: ", np.median(i))
    print("Standard Deviation: ", np.std(i))
    print("\n")
# plot relationships between variables
    
# deaths v.s. cases
plt.figure()
plt.scatter(cases, deaths)
plt.title("Total Deaths / 1M pop v.s. Total Cases / 1M pop")
plt.xlabel("Total Cases / 1M pop")
plt.ylabel("Total Deaths / 1M pop")
plt.show()
# calculate r^2
correlation_matrix = np.corrcoef(cases, deaths)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("R^2 = ", r_squared)
# cases v.s. tests
plt.figure()
plt.scatter(tests, cases)
plt.title("Total Cases / 1M pop v.s. Total Tests / 1M pop")
plt.xlabel("Total Tests / 1M pop")
plt.ylabel("Total Cases / 1M pop")
plt.show()
# calculate r^2
correlation_matrix = np.corrcoef(tests, cases)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("R^2 = ", r_squared)
# deaths v.s. beds
plt.figure()
plt.scatter(smoking, deaths)
plt.title("Total Deaths / 1M pop v.s. Smoking Prevalence")
plt.xlabel("Smoking Prevalence (% of Adult Population)")
plt.ylabel("Total Deaths / 1M pop")
plt.show()
# calculate r^2
correlation_matrix = np.corrcoef(tests, cases)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("R^2 = ", r_squared)
'''

# implement k-means clustering
from sklearn import preprocessing
# initialize the 6D list of attributes
attributes = []
for i in range(len(cases)):
    attributes.append([cases[i], deaths[i], tests[i], beds[i], UHC[i], smoking[i]])
# normalization
normalized_attributes = preprocessing.normalize(attributes)

# Elbow Plot
from sklearn.cluster import KMeans
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(normalized_attributes)
    kmeans.fit(normalized_attributes)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow Plot')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

# apply kmeans with k = 5 and a convergence criterion of 0.001
kmeans5 = KMeans(n_clusters=5, tol=0.001) 
clusters = kmeans5.fit_predict(normalized_attributes) 

# print out group 0 countries
group_cases = []
group_deaths = []
group_tests = []
group_beds = []
group_UHC = []
group_smoking = [] 

for i in range(len(clusters)):
    # iteratively consider 0, 1, 2, 3, and 4
    if clusters[i] == 4:
        group_cases.append(cases[i])
        group_deaths.append(deaths[i])
        group_tests.append(tests[i])
        group_beds.append(beds[i])
        group_UHC.append(UHC[i])
        group_smoking.append(smoking[i])
        print(countries[i])
        
group_attributes = [group_cases, group_deaths, group_tests, \
                    group_beds, group_UHC, group_smoking]

# print out corresponding summary statistics
for i in group_attributes:
    print("Mean: ", np.mean(i))
    print("SD: ", np.std(i))
    print("\n")