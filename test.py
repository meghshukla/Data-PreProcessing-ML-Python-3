#################################################################IMPLEMENTATION IN PYTHON 3#######################################################################

import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

directory='E:/IIT Bombay/IITB/Sem2/Machine Learning/Assignment 1/173310008/'

pickle_in=open(directory+'Models/dataPreProcess.pickle','rb')
dataPreProcess=pickle.load(pickle_in)
pickle_in=open(directory+'Models/FeaturesToDelete.pickle','rb')
FeaturesToDelete=pickle.load(pickle_in)
pickle_in=open(directory+'Models/dataConversion.pickle','rb')
dataConversion=pickle.load(pickle_in)

Mode=dataConversion['Mode']
Label=dataConversion['Label']

#Read Test CSV file
data_test=pd.read_csv(directory+'testSold.csv')
data_test.set_index('Id',inplace=True)

#Read Ground Truth CSV file
GroundTruth=pd.read_csv(directory+'gt.csv')
GroundTruth.set_index('Id',inplace=True)

Score={}                                                                        #Maintains scoring for each classifier


def PreProcess(Column,colName):
    '''
    Normalizes input dataset to values between 0 and 1
    Input : Unscaled real numbers
    Output: Normalized values between -1 to 1
    '''
    global dataPreProcess
    dummyColumn=[]
    Set[colName].fillna(dataPreProcess[colName][3],inplace=True)                                     #Replaces NA values with the mode of dataset
    if(dataPreProcess[colName][2]!=dataPreProcess[colName][1]):
        for i in Set.index:
            dummyColumn.append((2*(Column[i]-dataPreProcess[colName][0]))/float(dataPreProcess[colName][2]-dataPreProcess[colName][1]))
        return dummyColumn
    else:
        return Column
        

                                         #################NOMINAL TO ORDINAL DATA CONVERSION FUNCTION#########################
                                         
def labelToOrdinal(Column,colName):
    '''
    Converts labels such as Unf, Fin, Shed etc into numbers to give as input to classifier
    Input : Column containing String Data, name of column
    Output : String converted to numerical values
    '''
    global Mode
    global Label
    Set[colName].fillna(Mode[colName],inplace=True)                                     #Replaces NA values with the mode of dataset
    for i in Set.index:  
         Column[i]=Label[colName][Column[i]]           
    return Column
    


                                        #################NOMINAL TO ONE-HOT DATA CONVERSION FUNCTION####################    
                                        
                                        
def labelToOneHot(Column,colName):
    '''
    Converts Nominal variables into separate columns, each having 1 
    or 0 indicating presence/absence of that variable
    Input: Column with nominal variables
    Output: N columns for each of the N nominal labels in the column
    '''
    global Mode
    global Label
    Set[colName].fillna(Mode[colName],inplace=True)                                     #Replaces NA values with the mode of dataset      
    OneHotColumns={}
    for i in Label[colName]:
        name=colName+'_'+i
        OneHotColumns[name]=[]    
    for i in Set.index:
        for j in OneHotColumns.keys():
            OneHotColumns[j].append(0)
        name=colName+'_'+str(Column[i])
        OneHotColumns[name][-1]=1   
    for i in OneHotColumns.keys():
        Set[i]=OneHotColumns[i]      
    del Set[colName]
    
    
    
                                        #################NOMINAL TO BINARY DATA CONVERSION FUNCTION###################
                                        
def labelToBinary(Column,colName):
    '''
    This method handles Nominal data by first assigning the Labels some ordinal
    value such as 1,2 etc. It then converts these integers into binary format,
    with each number in binary contained in different columns.
    Eg : 6 --> 110 --> 1 , 1 , 0 in different columns. Similar to One-Hot,
    but reduces dimensionality when compared.
    More info at : https://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html
    
    Input: Column containing string data, name of column
    Output: Columns to hold each binary digit which represent a particular nominal label
    ''' 
    global Mode
    global Label   
    Set[colName].fillna(Mode[colName],inplace=True)                             #Replaces NA values with the mode of dataset
    binColumns={}          
    for i in range(len(bin(len(Label[colName])-1)[2:])):
        name=colName+'_'+str(i)
        binColumns[name]=[]                                                     #Generates columns depending on length of binary string
    for i in Set.index:
        p=bin(Label[colName][Column[i]])[2:]
        p=p[::-1]
        for k in range(len(bin(len(Label[colName])-1)[2:])):
            name=colName+'_'+str(k) 
            binColumns[name].append(0)                                          #Initializing to zero
        t=0
        for k in p:       
            name=colName+'_'+str(t)
            binColumns[name][-1]=k                                              #Assigns values to the columns based on ordinal value of label
            t+=1 
    for i in binColumns.keys():
        Set[i]=binColumns[i]                                                    #Adds the binary columns to the dataframe
    del Set[colName]                                                            #deletes the original column to avoid duplicity  
    
    
    
#########################################################################SCORING FUNCTION#################################################################################### 

def score(Output,classifier):
    accuracy=0
    total=0
    for i in GroundTruth.index:
        if(Output['SaleStatus'][i]==GroundTruth['SaleStatus'][i]):
            accuracy+=1
        total+=1
    Score[classifier]=(float(accuracy)/total)                  
    
      
#####################################################################CLASSIFICATION KERNELS##################################################################################


def LDAClass(InputSet_copy):
    '''
    Class prediction using Linear Discrimant Classification. 
    Loads pickle file saved by train.py to predict
    Outputs in a separate file called out_LDA.csv                    
    '''
    Id=InputSet_copy.index
    InputSet_copy=np.asmatrix(InputSet_copy)
    InputSet_copy=np.delete(InputSet_copy,FeaturesToDelete['LDA'],axis=1)       #These features were found by using recursive feature elimination in train.py
    
    pickle_LDA=open(directory+'Models/LDA.pickle','rb')
    lda=pickle.load(pickle_LDA)
    pred=lda.predict(InputSet_copy)
    Prediction={'Id':Id,'SaleStatus':pred}
    Prediction=pd.DataFrame(Prediction)
    Prediction.set_index('Id',inplace=True)
    score(Prediction,'LDA')
    Prediction.to_csv(directory+'Prediction/out_LDA.csv')
    #qda = QDA()
    #qda.fit(np.array(X_train),y_train)
    del lda

def SVMClass(InputSet_copy):
    '''
    Class prediction using SVM Classification. 
    Loads pickle file saved by train.py to predict.
    Outputs in a separate file called out_SVM.csv                    
    '''
    Id=InputSet_copy.index
    InputSet_copy=np.asmatrix(InputSet_copy)
    InputSet_copy=np.delete(InputSet_copy,FeaturesToDelete['SVM'],axis=1)       #These features were found by using recursive feature elimination in train.py
    
    #pca=PCA()
    #temp=pca.fit_transform(Set_)
    #PCA_Components_Set=pca.components_
    #InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCA_Components_Set))*np.transpose(np.asmatrix(InputSet_copy)))
    #del temp 
        
    pickle_SVM=open(directory+'Models/SVM.pickle','rb')
    SVM=pickle.load(pickle_SVM)  
    pred=SVM.predict(InputSet_copy)
    Prediction={'Id':Id,'SaleStatus':pred}
    Prediction=pd.DataFrame(Prediction)
    Prediction.set_index('Id',inplace=True)
    score(Prediction,'SVM')
    Prediction.to_csv(directory+'Prediction/out_SVM.csv')
        
    
def RFClass(InputSet_copy):
    '''
    Class prediction using Random Forest Classification. 
    Loads pickle file saved by train.py to predict.
    Outputs in a separate file called out_RFC.csv                           
    '''
    Id=InputSet_copy.index
    InputSet_copy=np.asmatrix(InputSet_copy)
    InputSet_copy=np.delete(InputSet_copy,FeaturesToDelete['RFC'],axis=1)       #These features were found by using recursive feature elimination in train.py
    
    pickle_in=open(directory+'Models/PCA_RFC.pickle','rb')                      #Needed to transform input into principal component space
    PCAcomponents=pickle.load(pickle_in)
    InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCAcomponents))*np.transpose(np.asmatrix(InputSet_copy)))

    pickle_in=open(directory+'Models/RFC.pickle','rb')
    RFC=pickle.load(pickle_in)
    pred=RFC.predict(InputSet_copy)
    Prediction={'Id':Id,'SaleStatus':pred}
    Prediction=pd.DataFrame(Prediction)
    Prediction.set_index('Id',inplace=True)
    score(Prediction,'RFC')
    Prediction.to_csv(directory+'Prediction/out_RFC.csv')  
    
          
def NNClass(InputSet_copy):
    '''
    Class prediction using Neural Network Classification. 
    Loads pickle file saved by train.py to predict.
    Outputs in a separate file called out_NN.csv                           
    '''
    Id=InputSet_copy.index
    InputSet_copy=np.asmatrix(InputSet_copy)
    InputSet_copy=np.delete(InputSet_copy,FeaturesToDelete['NN'],axis=1)       #These features were found by using recursive feature elimination in train.py       
    
    pickle_in=open(directory+'Models/PCA_MLP.pickle','rb')                      #Needed to transform input into principal component space
    PCAcomponents=pickle.load(pickle_in)
    InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCAcomponents))*np.transpose(np.asmatrix(InputSet_copy)))        

    pickle_in=open(directory+'Models/MLP.pickle','rb')
    NN=pickle.load(pickle_in)
    pred=NN.predict(InputSet_copy)
    Prediction={'Id':Id,'SaleStatus':pred}
    Prediction=pd.DataFrame(Prediction)
    Prediction.set_index('Id',inplace=True)
    score(Prediction,'NN')
    Prediction.to_csv(directory+'Prediction/out_MLP.csv')
    
                      
#########################################################################PREPROCESSING###########################################################################################


Set=data_test.copy()
Set=Set[Set.columns.sort_values()]

for i in Set.columns:
    if(type(Set[i][Set.index[1]])!=str):
        Set[i]=PreProcess(Set[i][:],i)
        
'''                                                                             
for i in Set.keys():                                                            #Converts nominal label to ordinal values        
    if(i!='SaleStatus' and  type(Set[i][Set.index[1]])==str):
        Set[i]=labelToOrdinal(Set[i][:],i)

for i in Set.keys():                                                           #Converts nominal labels to binary columnar values  
    if(i!='SaleStatus' and  type(Set[i]Set.index[1]])==str):
        labelToBinary(Set[i][:],i) 
'''        
for i in Set.keys():                                                           #Converts nominal labels to one-hot columnar values  
    if(i!='SaleStatus' and  type(Set[i][Set.index[1]])==str):
        labelToOneHot(Set[i][:],i)        
     
Set=pd.DataFrame(Set)                                                           #segregates the dataframe into input and output dataset
#InputSet=Set.drop('SaleStatus',axis=1).copy()
InputSet=Set.copy()

                  
###############################################################################################################################################################################

print('Evaluating LDA 1/4...')
LDAClass(InputSet.copy())
print('Evaluating SVM 2/4...')
SVMClass(InputSet.copy())
print('Evaluating RFC 3/4...')
RFClass(InputSet.copy())
print('Evaluating Neural Network 4/4...')
NNClass(InputSet.copy()) 

Max,Max_Label=[0,0]
for j in Score.keys():
    if(Score[j]>Max):
        Max=Score[j]
        Max_label=j
print('Maximum accuracy : '+str(Max)+', Classifier : '+Max_label)        
                                  
    
    

                  
                  