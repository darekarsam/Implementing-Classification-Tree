# Author:Sameer Darekar
#Title:Implementing 10 fold Cross Validation to evaluate accuracy
#data set location: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

import pandas as pd
import numpy as np
import math as mth
import copy as cp
import warnings
warnings.filterwarnings('ignore')

ht=0
class Node(object):
  def __init__(self):
    self.dataset= None
    self.predictDataset=None
    self.gini=None
    self.entropy=None
    self.ht=None
    self.left =None
    self.right=None
    self.leafFlag=None
    self.split=None
  def getLChild(self):
    return self.left
  def getRChild(self):
    return self.right


""" #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
"""

# function to read Dataset
def getData():

  data=pd.read_csv('C:\\Users\\samee\\OneDrive\\Assignments\\DM Assignment2\\4\\brstcancer.csv')
  data.columns=["ID","clump","size","shape","adhesion","scSize","bNuclei","chromatin","nNuclei","mitosis","ClassOfT"]
  return data

def calcGiniParent(treeNode,udata,row):
  gini=0
  for i in range(0,len(udata)):
    temp=float(sum(treeNode.dataset[row]==udata[i]))/len(treeNode.dataset[row])
    gini=gini+temp*temp
  gini=1-gini
  return gini

def calcGiniOfSplit(dataset,threshold,col,flag):
  if flag==1:
    #print dataset
    totalSplit1=len(dataset[dataset["ClassOfT"]<=threshold])
    totalSplit2=len(dataset[dataset["ClassOfT"]>threshold])
  else:
    totalSplit1=len(dataset[dataset[col]<=threshold])
    totalSplit2=len(dataset[dataset[col]>threshold])
  c00=len(dataset[(dataset[col]<=threshold)&(dataset["ClassOfT"]==2)])
  c10=len(dataset[(dataset[col]<=threshold)&(dataset["ClassOfT"]==4)])
  c01=len(dataset[(dataset[col]>threshold)&(dataset["ClassOfT"]==2)])
  c11=len(dataset[(dataset[col]>threshold)&(dataset["ClassOfT"]==4)])
  total=totalSplit1+totalSplit2
  col1=c00+c10
  col2=c01+c11
  if col1==0 or col2==0:
    return 1
  giniSplit1=((float(c00)/col1)*(float(c00)/col1))+((float(c10)/col1)*(float(c10)/col1))
  giniSplit1=1-giniSplit1
  giniSplit2=((float(c01)/col2)*(float(c01)/col2))+((float(c11)/col2)*(float(c11)/col2))
  giniSplit2=1-giniSplit2
  giniOfSplit=((float(col1)/(col1+col2))*giniSplit1)+((float(col2)/(col1+col2))*giniSplit2)
  return giniOfSplit

def calcInfoGain(treeNode,threshold,col,flag):
  dataset=treeNode.dataset
  if flag==1:
    totalSplit1=len(dataset[dataset["ClassOfT"]<=threshold])
    totalSplit2=len(dataset[dataset["ClassOfT"]>threshold])
  else:
    totalSplit1=len(dataset[dataset[col]<=threshold])
    totalSplit2=len(dataset[dataset[col]>threshold])
  c00=len(dataset[(dataset[col]<=threshold)&(dataset["ClassOfT"]==2)])
  c10=len(dataset[(dataset[col]<=threshold)&(dataset["ClassOfT"]==4)])
  c01=len(dataset[(dataset[col]>threshold)&(dataset["ClassOfT"]==2)])
  c11=len(dataset[(dataset[col]>threshold)&(dataset["ClassOfT"]==4)])
  total=totalSplit1+totalSplit2
  col1=c00+c10
  col2=c01+c11
  if col1==0 or col2==0:
    return 1
  if c00==0:
    entropyOfSplit1=0
  else:
    entropyOfSplit1=(float(c00)/col1)*mth.log((float(c00)/col1),2)
  entropyOfSplit1=-1*entropyOfSplit1
  if c10!=0:
    entropyOfSplit1=entropyOfSplit1- ((float(c10)/col1)*mth.log((float(c10)/col1)))
  if c01==0:
    entropyOfSplit2=0
  else:
    entropyOfSplit2=(float(c01)/col2)*mth.log((float(c01)/col2),2)
  entropyOfSplit2=-1*entropyOfSplit2
  if c11!=0:
    entropyOfSplit2=entropyOfSplit2- ((float(c11)/col2)*mth.log((float(c11)/col2)))
  entropyOfSplit=((float(col1)/(col1+col2))*entropyOfSplit1)+((float(col2)/(col1+col2))*entropyOfSplit2)
  gain=treeNode.entropy-entropyOfSplit
  return gain

def decideSplitAttribute(treeNode,criteria,cols):

  giniCols=[]
  for col in cols:
    flag=0
    treeNode.dataset.sort_values([col,"ClassOfT"],inplace=True)
    uniqNo=treeNode.dataset[col].unique()
    countUniq=len(uniqNo)
    if countUniq<=10:   #if Categorical variable
      uniq=uniqNo[0:len(uniqNo)-1] #Remove last element of uniq
      if countUniq==1:  #if uniq count =1 split on basis of class variable
        continue
      giniOfCol=[]
      for i in uniq:  # for GINI
        if treeNode.entropy==None:
          giniOfCol.append([i,calcGiniOfSplit(treeNode.dataset,i,col,flag),col,flag]) #get in list(split threshold value, gini)
            #getting the index of minimum of the lot

        else:    #for Information Gain
          giniOfCol.append([i,calcInfoGain(treeNode,i,col,flag),col,flag])
      if treeNode.entropy==None:
        b=np.asarray(giniOfCol)
        li=b.argmin(axis=0)
        giniCols.append(giniOfCol[li[1]]) # store min gini
      else:
        b=np.asarray(giniOfCol)  #getting the index of maximum of the lot
        li=b.argmax(axis=0)
        giniCols.append(giniOfCol[li[1]])
    if treeNode.entropy==None:
      b=np.asarray(giniCols)
      li=b.argmin(axis=0)
    else:
      b=np.asarray(giniCols)
      li=b.argmax(axis=0)
    returnVal=giniCols[li[1]]
    returnVal=returnVal[-1:]
  return giniCols[li[1]]   #gives the (threshold,Gini/IG,column name) is criteria=1 else gives(threshold,entropy,column name)


def calcEntropyOfParent(treeNode,udata,row):
  entropy=0
  for i in range(0,len(udata)):
    temp=float(sum(treeNode.dataset[row]==udata[i]))/len(treeNode.dataset[row])
    logar=(mth.log(temp,2))
    entropy=entropy-(temp*logar)
  treeNode.entropy=entropy
  return entropy

#function to calculate split criteria i.e. GINI or information Gain
def calcSplitCriteria(treeNode,criteria,cols):
  udata=treeNode.dataset["ClassOfT"].unique()
  if criteria=='1':  # calculate GINI
    treeNode.gini=calcGiniParent(treeNode,udata,"ClassOfT") #gini of Node calculated
  else: #calculate information Gain
    treeNode.gain=calcEntropyOfParent(treeNode,udata,"ClassOfT")
  attribute=decideSplitAttribute(treeNode,criteria,cols)
  return attribute

def checkLeaf(treeNode):
  if len(treeNode.dataset["ClassOfT"].unique())==1:
    return True
  return False

def getMaxCountOFClass(treeNode):
  uniq=treeNode.dataset["ClassOfT"].unique().tolist()
  leafValCount=[]
  for col in uniq:
    leafValCount.append([len(treeNode.dataset[treeNode.dataset["ClassOfT"]==col]),col])
  b=np.asarray(leafValCount)
  li=b.argmax(axis=0)
  return b[li[0]]

# function to build tree
def buildTree(treeNode,criteria,cols,depth):
  treeNode.ht=depth+1
  if (checkLeaf(treeNode)):
    treeNode.left=None
    treeNode.right=None
    treeNode.leafFlag=1
    classVal=treeNode.dataset["ClassOfT"].unique()
    return treeNode
  treeNode.leafFlag=0
  splitCo=calcSplitCriteria(treeNode,criteria,cols)
  treeNode.split=splitCo
  leftc=Node()
  rightc=Node()
  treeNode.dataset.sort_values([splitCo[2],"ClassOfT"],inplace=True)
  colOfSplit=treeNode.split[2]
  threshold=treeNode.split[0]
  leftc.dataset=treeNode.dataset[treeNode.dataset[colOfSplit]<=threshold]
  rightc.dataset=treeNode.dataset[treeNode.dataset[colOfSplit]>threshold]
  parentCount=getMaxCountOFClass(treeNode)
  leftCount=getMaxCountOFClass(leftc)
  rightCount=getMaxCountOFClass(rightc)
  children=(leftCount[0]+rightCount[0])
  treeNode.left=leftc
  treeNode.right=rightc
  if((children-parentCount[0])<=0):
    treeNode.leafFlag=1
    treeNode.left=None
    treeNode.right=None
    return treeNode
  treeNode.left=buildTree(treeNode.left,criteria,cols,treeNode.ht)
  treeNode.right=buildTree(treeNode.right,criteria,cols,treeNode.ht)
  if treeNode.leafFlag==1:
    treeNode.left=None
    treeNode.right=None
  return treeNode


def displayTree(treeNode):
  if treeNode.right!=None:
    displayTree(treeNode.right)
  if(treeNode.leafFlag==1):
    classVal=treeNode.dataset["ClassOfT"].unique()
    print "|   "*treeNode.ht,"Leaf Node with Class: ",str(classVal)
  else:
    print "|   "*treeNode.ht,str(treeNode.split)
  if(treeNode.left!=None):
    displayTree(treeNode.left)

def displayPreOrder(treeNode):
  if treeNode.leafFlag==1:
    classVal=treeNode.dataset["ClassOfT"].unique()
    print"Leaf Node with Class: ",str(classVal)
  else:
    print treeNode.split,"|"
  if(treeNode.left!=None):
    displayPreOrder(treeNode.left)
  if treeNode.right!=None:
    displayPreOrder(treeNode.right)


def validateTestData(treeNode):
  if treeNode.leafFlag==1:
    leafVal=treeNode.dataset["ClassOfT"].unique()[0]
    valsToReplace={0:leafVal}
    treeNode.predictDataset['ClassOfT']=treeNode.predictDataset['ClassOfT'].map(valsToReplace) #replace values of class variable to leaf value
    return treeNode
  threshold=treeNode.split[0]
  colOfSplit=treeNode.split[2]
  treeNode.getLChild().predictDataset=treeNode.predictDataset[treeNode.predictDataset[colOfSplit]<=threshold]
  treeNode.getRChild().predictDataset=treeNode.predictDataset[treeNode.predictDataset[colOfSplit]>threshold]
  treeNode.left=validateTestData(treeNode.left)
  treeNode.right=validateTestData(treeNode.right)
  leftc=treeNode.left
  rightc=treeNode.right
  frames=[leftc.predictDataset,rightc.predictDataset]
  treeNode.predictDataset=pd.concat(frames)
  return treeNode

def perform10FoldValidation(dataset,criteria,cols):
  start=0
  stats=[]
  for i in range (1,11):
    print"In Proces for Fold ",i
    if i!=10:
      end=(len(dataset)/10)*i
      testDataset=dataset[start:end]
      trainingDataset=dataset.drop(dataset.index[start:end])
    else:
      testDataset=dataset[start:]
      trainingDataset=dataset.drop(dataset.index[start:])
    root=Node()
    root.dataset=trainingDataset
    root=buildTree(root,criteria,cols,-1)
    valsToReplace={2:0,4:0}
    testDataset.sort_values(["ID"], inplace=True)
    predictDataset=cp.deepcopy(testDataset)
    predictDataset['ClassOfT']=predictDataset['ClassOfT'].map(valsToReplace) #replace values of class variable to 0
    root.predictDataset=predictDataset
    root=validateTestData(root)
    root.predictDataset.sort_values(["ID"],inplace=True)
    start=end
    y_true=testDataset["ClassOfT"].tolist()
    y_pred=root.predictDataset["ClassOfT"].tolist()
    diff=0
    trueCount=0
    for j in range(0,len(y_pred)):
      if(y_pred[j]!=y_true[j]):
        diff=diff+1
    trueCount=len(y_pred)-diff
    #confusionMatrix= pd.crosstab(y_true,y_pred,rownames=['True'], colnames=['Predicted'], margins=True)
    correctPrediction=trueCount
    totalNoOfPrediction=len(y_pred)
    print "For Fold {0}: {1} predicted correctly out of {2}".format(i,correctPrediction,totalNoOfPrediction)
    stats.append([correctPrediction,totalNoOfPrediction])
  allStats=np.asarray(stats)
  allStats=np.sum(allStats,axis=0)
  accuracy=float(allStats[0])/allStats[1]
  print "Average Accuracy is :",accuracy
  return

dataset=getData()
criteria=1
while(criteria==1 or criteria==2):
  criteria=raw_input("Enter 1 for GINI or 2 for Information Gain : ") #criteria=1 for GINI and 2 for Information Gain
columns=["ID","clump","size","shape","adhesion","scSize","bNuclei","chromatin","nNuclei","mitosis","ClassOfT"]
cols=columns[1:10]
#dividing the dataset into 10 parts
perform10FoldValidation(dataset,criteria,cols)
# print"Displaying Tree in Pre Order"
# print" "
# print "For Non Leaf Node, The Tree Node display is [Threshold,GINI,Column Name,Leaf Flag=0 if non leaf]"
# print "for Leaf Node, it displays the Class"
# displayPreOrder(root)
# print "For Non Leaf Node, The Tree Node display is [Threshold,GINI,Column Name,Leaf Flag=0 if non leaf]"
# print "for Leaf Node, it displays the Class"
# print"Levels"
# print"0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   "
# displayTree(root)
# print"0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   "
# print"Levels"




"""
#output for Gini
Enter 1 for GINI or 2 for Information Gain : 1
In Proces for Fold  1
For Fold 1: 38 predicted correctly out of 68
In Proces for Fold  2
For Fold 2: 46 predicted correctly out of 68
In Proces for Fold  3
For Fold 3: 41 predicted correctly out of 68
In Proces for Fold  4
For Fold 4: 31 predicted correctly out of 68
In Proces for Fold  5
For Fold 5: 41 predicted correctly out of 68
In Proces for Fold  6
For Fold 6: 51 predicted correctly out of 68
In Proces for Fold  7
For Fold 7: 49 predicted correctly out of 68
In Proces for Fold  8
For Fold 8: 58 predicted correctly out of 68
In Proces for Fold  9
For Fold 9: 46 predicted correctly out of 68
In Proces for Fold  10
For Fold 10: 57 predicted correctly out of 70
Average Accuracy is : 0.671554252199


#output for Information Gain
Enter 1 for GINI or 2 for Information Gain : 2
In Proces for Fold  1
For Fold 1: 37 predicted correctly out of 68
In Proces for Fold  2
For Fold 2: 48 predicted correctly out of 68
In Proces for Fold  3
For Fold 3: 41 predicted correctly out of 68
In Proces for Fold  4
For Fold 4: 31 predicted correctly out of 68
In Proces for Fold  5
For Fold 5: 34 predicted correctly out of 68
In Proces for Fold  6
For Fold 6: 52 predicted correctly out of 68
In Proces for Fold  7
For Fold 7: 48 predicted correctly out of 68
In Proces for Fold  8
For Fold 8: 58 predicted correctly out of 68
In Proces for Fold  9
For Fold 9: 46 predicted correctly out of 68
In Proces for Fold  10
For Fold 10: 57 predicted correctly out of 70
Average Accuracy is : 0.66275659824

"""