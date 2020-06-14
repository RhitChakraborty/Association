import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
# print(dataset.head())
# items=dataset.unique()
# print(items)



transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#training the model
#from mlxtend.frequent_patterns import apriori,association_rules
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
result=list(rules)
# print(result)
result_detailed = []
for i in range(len(result)):
    result_detailed.append('Rule:\t' + str(result[i][0]) +
                            '\nSupport:\t' + str(result[i][1]) +
                            '\nConfidence:\t' + str(result[i][2][0][2]) +
                            '\nLift:\t' + str(result[i][2][0][3]))
for j in result_detailed:
    print(j)