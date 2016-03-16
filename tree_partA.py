# Author:Sameer Darekar
#Title:Implementing Classification trees and evaluating their accuracy
#data set location: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

import pandas as pd
import numpy as np
import math as mth
import warnings
warnings.filterwarnings('ignore')

ht=0
class Node(object):
  def __init__(self):
    self.dataset= None
    self.gini=None
    self.entropy=None
    self.ht=None
    self.left =None
    self.right=None
    self.leafFlag=None
    self.split=None
    def getLchild(self):
      return self.left
    def getRchild(self):
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
        """flag=1
        uniq=treeNode.dataset["ClassOfT"].unique()
        uniq=uniq[0:len(uniq)-1]"""
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
  return giniCols[li[1]]   #gives the (threshold,Gini,column name) is criteria=1 else gives(threshold,entropy,column name)


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

# function to build tree
def buildTree(treeNode,criteria,cols,depth):
  treeNode.ht=depth+1
  if (checkLeaf(treeNode)):
    treeNode.left=None
    treeNode.right=None
    treeNode.leafFlag=1
    classVal=treeNode.dataset["ClassOfT"].unique()
    #print "Leaf Node with class: {0}".format(classVal)
    #print treeNode.dataset.head()
    return treeNode
  treeNode.leafFlag=0
  splitCo=calcSplitCriteria(treeNode,criteria,cols)
  treeNode.split=splitCo
  #print treeNode.split
  leftc=Node()
  rightc=Node()
  treeNode.dataset.sort_values([splitCo[2],"ClassOfT"],inplace=True)
  colOfSplit=treeNode.split[2]
  threshold=treeNode.split[0]
  leftc.dataset=treeNode.dataset[treeNode.dataset[colOfSplit]<=threshold]
  rightc.dataset=treeNode.dataset[treeNode.dataset[colOfSplit]>threshold]
  treeNode.left=leftc
  treeNode.right=rightc
  treeNode.left=buildTree(treeNode.left,criteria,cols,treeNode.ht)
  treeNode.right=buildTree(treeNode.right,criteria,cols,treeNode.ht)
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


dataset=getData()
criteria=1
while(criteria==1 or criteria==2):
  criteria=raw_input("Enter 1 for GINI or 2 for Information Gain : ") #criteria=1 for GINI and 2 for Information Gain
columns=["ID","clump","size","shape","adhesion","scSize","bNuclei","chromatin","nNuclei","mitosis","ClassOfT"]
cols=columns[1:10]
root=Node()
root.dataset=dataset
print "Building Tree"
root=buildTree(root,criteria,cols,-1)
print" "
print"Displaying Tree in Pre Order"
print" "
print "For Non Leaf Node, The Tree Node display is [Threshold,GINI,Column Name,Leaf Flag=0 if non leaf]"
print "for Leaf Node, it displays the Class"
displayPreOrder(root)
print "For Non Leaf Node, The Tree Node display is [Threshold,GINI,Column Name,Leaf Flag=0 if non leaf]"
print "for Leaf Node, it displays the Class"
print " "
print "Natural Display level 0 is root and level 1 are its children and so on"
print"Levels"
print"0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   "
displayTree(root)
print"0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   "
print"Levels"



"""
#Output
Enter 1 for GINI or 2 for Information Gain : 1
Building Tree
 
Displaying Tree in Pre Order
 
For Non Leaf Node, The Tree Node display is [Threshold,GINI/Information Gain,Column Name,Leaf Flag=0 if non leaf]
for Leaf Node, it displays the Class
[2, 0.1296352482557748, 'size', 0] |
[5, 0.027884294031767246, 'bNuclei', 0] |
[6, 0.015599506160885067, 'clump', 0] |
[8, 0.004938210942682475, 'nNuclei', 0] |
[4, 0.004342431761786601, 'bNuclei', 0] |
Leaf Node with Class:  [2]
[1, 0.0, 'scSize', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[2, 0.0, 'shape', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[1, 0.0, 'clump', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[2, 0.16800113899077276, 'shape', 0] |
[5, 0.08237986270022897, 'clump', 0] |
[6, 0.0, 'adhesion', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
[4, 0.12973781704189233, 'size', 0] |
[2, 0.2632275132275132, 'bNuclei', 0] |
[3, 0.12987012987012997, 'adhesion', 0] |
[6, 0.0, 'nNuclei', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
[6, 0.20009350163627862, 'clump', 0] |
[5, 0.3359683794466404, 'clump', 0] |
[5, 0.28593628593628584, 'bNuclei', 0] |
[3, 0.18518518518518512, 'size', 0] |
[3, 0.0, 'scSize', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [2]
[3, 0.07692307692307693, 'chromatin', 0] |
[3, 0.0, 'size', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
[7, 0.05734767025089606, 'bNuclei', 0] |
[4, 0.1111111111111111, 'adhesion', 0] |
Leaf Node with Class:  [4]
[4, 0.0, 'shape', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
[1, 0.030232513414235423, 'adhesion', 0] |
[6, 0.18181818181818182, 'clump', 0] |
[6, 0.0, 'shape', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[2, 0.011815496478073196, 'nNuclei', 0] |
[1, 0.05925925925925923, 'nNuclei', 0] |
Leaf Node with Class:  [4]
[6, 0.0, 'clump', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]

Natural Display level 0 is root and level 1 are its children and so on
For Non Leaf Node, The Tree Node display is [Threshold,GINI/Information Gain,Column Name,Leaf Flag=0 if non leaf]
for Leaf Node, it displays the Class
Levels
0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   
|   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [2, 0.011815496478073196, 'nNuclei', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [6, 0.0, 'clump', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [1, 0.05925925925925923, 'nNuclei', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |    [1, 0.030232513414235423, 'adhesion', 0]
|   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [6, 0.18181818181818182, 'clump', 0]
|   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [6, 0.0, 'shape', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |    [4, 0.12973781704189233, 'size', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |    [7, 0.05734767025089606, 'bNuclei', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |   |    [4, 0.0, 'shape', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |    [4, 0.1111111111111111, 'adhesion', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [6, 0.20009350163627862, 'clump', 0]
|   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [5, 0.3359683794466404, 'clump', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |   |    [3, 0.07692307692307693, 'chromatin', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |   |   |    [3, 0.0, 'size', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [5, 0.28593628593628584, 'bNuclei', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |   |    [3, 0.18518518518518512, 'size', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |   |   |    [3, 0.0, 'scSize', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |    [2, 0.2632275132275132, 'bNuclei', 0]
|   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [3, 0.12987012987012997, 'adhesion', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |    [6, 0.0, 'nNuclei', 0]
|   |   |   |   |   |    Leaf Node with Class:  [2]
|    [2, 0.16800113899077276, 'shape', 0]
|   |   |    Leaf Node with Class:  [4]
|   |    [5, 0.08237986270022897, 'clump', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [6, 0.0, 'adhesion', 0]
|   |   |   |    Leaf Node with Class:  [2]
 [2, 0.1296352482557748, 'size', 0]
|   |   |    Leaf Node with Class:  [4]
|   |    [1, 0.0, 'clump', 0]
|   |   |    Leaf Node with Class:  [2]
|    [5, 0.027884294031767246, 'bNuclei', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [2, 0.0, 'shape', 0]
|   |   |   |    Leaf Node with Class:  [2]
|   |    [6, 0.015599506160885067, 'clump', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [8, 0.004938210942682475, 'nNuclei', 0]
|   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [1, 0.0, 'scSize', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [4, 0.004342431761786601, 'bNuclei', 0]
|   |   |   |   |    Leaf Node with Class:  [2]
0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   
Levels

Enter 1 for GINI or 2 for Information Gain : 2
Building Tree
 
Displaying Tree in Pre Order
 
For Non Leaf Node, The Tree Node display is [Threshold,GINI/Information Gain,Column Name,Leaf Flag=0 if non leaf]
for Leaf Node, it displays the Class
[2, 0.6393767337929486, 'size', 0] |
[3, 0.11034524797756415, 'bNuclei', 0] |
[7, 0.039914378821606984, 'clump', 0] |
Leaf Node with Class:  [2]
[1, 0.9182958340544896, 'shape', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[3, 0.6836429189226791, 'clump', 0] |
Leaf Node with Class:  [2]
[2, 0.36783122488836323, 'chromatin', 0] |
[3, 1.0, 'adhesion', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[4, 0.23325687050338628, 'size', 0] |
[3, 0.4067221036960744, 'bNuclei', 0] |
[1, 0.3500449669081558, 'nNuclei', 0] |
Leaf Node with Class:  [2]
[4, 0.4367687757467055, 'clump', 0] |
[6, 0.4689955935892812, 'adhesion', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[2, 0.3191414231465465, 'bNuclei', 0] |
[3, 0.40758838657040464, 'adhesion', 0] |
[4, 0.49177205106717164, 'shape', 0] |
[2, 0.9182958340544896, 'scSize', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
[4, 0.15707075143909316, 'shape', 0] |
[5, 0.25661308486621565, 'adhesion', 0] |
[4, 0.25422731388396613, 'adhesion', 0] |
[5, 0.3886235648551747, 'bNuclei', 0] |
[4, 0.5614321901739298, 'bNuclei', 0] |
Leaf Node with Class:  [4]
[2, 0.8112781244591328, 'clump', 0] |
Leaf Node with Class:  [4]
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
[8, 0.8112781244591328, 'clump', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
[4, 0.04099283325438717, 'chromatin', 0] |
[6, 0.13163377791989567, 'clump', 0] |
[8, 0.3271097341407061, 'bNuclei', 0] |
[6, 0.6093128704690486, 'shape', 0] |
Leaf Node with Class:  [4]
[9, 0.8112781244591328, 'size', 0] |
Leaf Node with Class:  [2]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
Leaf Node with Class:  [4]
For Non Leaf Node, The Tree Node display is [Threshold,GINI,Column Name,Leaf Flag=0 if non leaf]
for Leaf Node, it displays the Class
Levels
0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   
|   |   |    Leaf Node with Class:  [4]
|   |    [4, 0.04099283325438717, 'chromatin', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [6, 0.13163377791989567, 'clump', 0]
|   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [8, 0.3271097341407061, 'bNuclei', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [9, 0.8112781244591328, 'size', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [6, 0.6093128704690486, 'shape', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|    [4, 0.23325687050338628, 'size', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [4, 0.15707075143909316, 'shape', 0]
|   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |    [5, 0.25661308486621565, 'adhesion', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [8, 0.8112781244591328, 'clump', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |    [4, 0.25422731388396613, 'adhesion', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [5, 0.3886235648551747, 'bNuclei', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |   |   |    [2, 0.8112781244591328, 'clump', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |   |    [4, 0.5614321901739298, 'bNuclei', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |    [3, 0.4067221036960744, 'bNuclei', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |    [2, 0.3191414231465465, 'bNuclei', 0]
|   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |    [3, 0.40758838657040464, 'adhesion', 0]
|   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |   |   |   |    [4, 0.49177205106717164, 'shape', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |   |   |   |    [2, 0.9182958340544896, 'scSize', 0]
|   |   |   |   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |    [4, 0.4367687757467055, 'clump', 0]
|   |   |   |   |   |    Leaf Node with Class:  [4]
|   |   |   |   |    [6, 0.4689955935892812, 'adhesion', 0]
|   |   |   |   |   |    Leaf Node with Class:  [2]
|   |   |    [1, 0.3500449669081558, 'nNuclei', 0]
|   |   |   |    Leaf Node with Class:  [2]
 [2, 0.6393767337929486, 'size', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [2, 0.36783122488836323, 'chromatin', 0]
|   |   |   |   |    Leaf Node with Class:  [2]
|   |   |   |    [3, 1.0, 'adhesion', 0]
|   |   |   |   |    Leaf Node with Class:  [4]
|   |    [3, 0.6836429189226791, 'clump', 0]
|   |   |    Leaf Node with Class:  [2]
|    [3, 0.11034524797756415, 'bNuclei', 0]
|   |   |   |    Leaf Node with Class:  [4]
|   |   |    [1, 0.9182958340544896, 'shape', 0]
|   |   |   |    Leaf Node with Class:  [2]
|   |    [7, 0.039914378821606984, 'clump', 0]
|   |   |    Leaf Node with Class:  [2]
0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   
Levels



"""