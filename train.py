#################################################################IMPLEMENTATION IN PYTHON 3#######################################################################

import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import stats
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
warnings.filterwarnings(action='ignore', category= DataConversionWarning)
warnings.filterwarnings(action='ignore', category= UserWarning)
warnings.filterwarnings(action='ignore', category= ConvergenceWarning)
warnings.filterwarnings(action='ignore', category= FutureWarning)
warnings.filterwarnings(action='ignore', category= RuntimeWarning)

directory='E:/IIT Bombay/IITB/Sem2/Machine Learning/Assignment 1/173310008/'

#Read Train CSV file
data_train=pd.read_csv(directory+'trainSold.csv')
data_train.set_index('Id',inplace=True)
#ID : 1460 is last entry in train.csv

#Read Test CSV, needed to compute PCA over Train+Test data
data_test=pd.read_csv(directory+'testSold.csv')
data_test.set_index('Id',inplace=True)

dataPreProcess={}
FeaturesToDelete={}
dataConversion={}
Mode={}
Label={}


def PreProcess(Column,colName):
    '''
    Normalizes input dataset to values between 0 and 1
    Input : Unscaled real numbers
    Output: Normalized values between -1 to 1
    '''
    global dataPreProcess
    dummyColumn=[]
    mu=np.mean(Column)
    Min=np.min(Column)
    Max=np.max(Column)
    Mode=stats.mode(Column)[0][0]
    dataPreProcess[colName]=[mu,Min,Max,Mode]
    Set[colName].fillna(Mode,inplace=True)                                     #Replaces NA values with the mode of dataset
    if(Max!=Min):
        for i in Set.index:
            dummyColumn.append((2*(Column[i]-mu))/float(Max-Min))
        return dummyColumn
    else:
        return Column    
        
                
    


                                         #################NOMINAL TO ORDINAL DATA CONVERSION FUNCTION################################
                                         
def labelToOrdinal(Column,colName):
    '''
    Converts labels such as Unf, Fin, Shed etc into numbers to give as input to classifier
    Input : Column containing String Data, name of column
    Output : String converted to numerical values
    '''
    global Mode
    global Label
    Labels={}
    j=-1
    Column=Column.dropna()
    Mode[colName]=stats.mode(Column)[0][0]
    Set[colName].fillna(Mode[colName],inplace=True)                                     #Replaces NA values with the mode of dataset
    Column=Set[colName]
    for i in Set.index:  
         #indexing via Columns starts at 1 and not 0. Column[1] gives first entry in the column
         if Column[i] not in Labels.keys():
             j+=1
             Labels[Column[i]]=j
         Column[i]=Labels[Column[i]]
    Label[colName]=Labels                
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
    Labels=[]
    Column=Column.dropna()
    Mode[colName]=stats.mode(Column)[0][0]
    Set[colName].fillna(Mode[colName],inplace=True)                                     #Replaces NA values with the mode of dataset
    Column=Set[colName]
    for i in Set.index:
        if Column[i] not in Labels:                                        
            Labels.append(Column[i])        
    OneHotColumns={}
    for i in Labels:
        name=colName+'_'+i
        OneHotColumns[name]=[]    
    for i in Set.index:
        for j in OneHotColumns.keys():
            OneHotColumns[j].append(0)
        name=colName+'_'+str(Column[i])
        OneHotColumns[name][-1]=1   
    for i in OneHotColumns.keys():
        Set[i]=OneHotColumns[i]  
    Label[colName]=Labels     
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
    Labels={}
    j=-1
    Column=Column.dropna()
    Mode[colName]=stats.mode(Column)[0][0]
    Set[colName].fillna(Mode[colName],inplace=True)                                     #Replaces NA values with the mode of dataset
    Column=Set[colName]
    for i in Set.index:  
         if Column[i] not in Labels.keys():
             j+=1
             Labels[Column[i]]=j                                                #Assigns ordinal value to particular label       
    binColumns={}          
    for i in range(len(bin(j)[2:])):
        name=colName+'_'+str(i)
        binColumns[name]=[]                                                     #Generates columns depending on length of binary string
    for i in Set.index:
        p=bin(Labels[Column[i]])[2:]
        p=p[::-1]
        for k in range(len(bin(j)[2:])):
            name=colName+'_'+str(k) 
            binColumns[name].append(0)                                          #Initializing to zero
        t=0
        for k in p:       
            name=colName+'_'+str(t)
            binColumns[name][-1]=int(k)                                          #Assigns values to the columns based on ordinal value of label
            t+=1 
    for i in binColumns.keys():
        Set[i]=binColumns[i]                                                    #Adds the binary columns to the dataframe
    Label[colName]=Labels
    del Set[colName]                                                            #deletes the original column to avoid duplicity  
   
    
      
#####################################################################CLASSIFICATION KERNELS##################################################################################


def LDAClass(InputSet_copy,OutputSet_copy):
    '''
    Performs Linear Discrimant Classification. Function is divided into 4 parts.
    1st part: Finds the best features to perform classification on.
              Uses recursive feature elimination. Only those columns are retained
              in the Input Data which will be used for classification.
    2nd part: Hyperparameter tuning using Grid Search. Results are stored in a
              separate .csv file.
    3rd part: Comparison of hyperparamters by effect on cross validation score.
    4th part: Saving model as pickle file                        
    '''
    global FeaturesToDelete
    InputSet_copy=np.asmatrix(InputSet_copy)
    lda = LDA()                                                                     
    selector = RFECV(lda)
    selector = selector.fit(InputSet_copy,OutputSet_copy)
    SelectedFeatures=selector.support_ 
    ColumnsToDelete=[]
    for i in range(len(SelectedFeatures)):
        if(SelectedFeatures[i]==False):
            ColumnsToDelete.append(i)
    FeaturesToDelete['LDA']=ColumnsToDelete[:]               
    InputSet_copy=np.delete(InputSet_copy,ColumnsToDelete,axis=1)  
    del lda
    del selector

    lda=LDA(n_components=2)
    parameters = {'solver':('svd', 'lsqr')}
    LD=GridSearchCV(lda,parameters)
    LD.fit(np.array(InputSet_copy),OutputSet_copy)
    a=pd.DataFrame(LD.cv_results_)
    a.to_csv(directory+'Hyperparamter Data/LinearDiscrimantAnalysis.csv')
    #qda = QDA()
    #qda.fit(np.array(X_train),y_train)
    
    plt.figure()
    plt.scatter(0,a['mean_test_score'][0],c='y',marker='x')
    plt.scatter(1,a['mean_test_score'][1],c='y',marker='o')
    plt.xlim(-1,2)
    plt.ylim(0,1)
    plt.xticks([0,1],['SVD','LSQR'])
    plt.xlabel('Solver : SVD, LSQR')
    plt.ylabel('CV Test score')
    plt.title('Comparison of LDA solvers on score')
    #plt.show()
    del lda,LD

    #Saving the model as pickle
    lda=LDA(solver='svd')
    lda.fit(InputSet_copy,OutputSet_copy)
    pickle_LDA=open(directory+'Models/LDA.pickle','wb')
    pickle.dump(lda,pickle_LDA)
    pickle_LDA.close()
    


def SVMClass(InputSet_copy,OutputSet_copy):
    '''
    Performs SVM classification on Input. Function is divided into 5 parts.
    1st part: Finds the best features to perform classification on.
              Uses recursive feature elimination. Only those columns are retained
              in the Input Data which will be used for classification.
    2nd part: Use of principal component analysis. Decorrelates the input data
              by projecting input space into feature space where axes are 
              perpendicular to each other. Reduces overfitting to some extent.
              Currently PCA code commented out.
    3rd part: Hyperparameter tuning using Grid Search. Results are stored in a
              separate .csv file.    
    4th part: Comparison of hyperparamters by effect on cross validation score 
    5th part: Saving model as pickle file                         
    '''
    global FeaturesToDelete
    InputSet_copy=np.asmatrix(InputSet_copy)
    #OutputSet_copy=np.asmatrix(OutputSet_copy)
    OutputSet_copy=np.ravel(OutputSet_copy)
    svc = svm.SVC(kernel='linear')
    selector = RFECV(svc)
    selector = selector.fit(InputSet_copy,OutputSet_copy)
    SelectedFeatures=selector.support_ 
    ColumnsToDelete=[]
    for i in range(len(SelectedFeatures)):
        if(SelectedFeatures[i]==False):
            ColumnsToDelete.append(i)
    FeaturesToDelete['SVM']=ColumnsToDelete[:]           
    Set_=Set[:].copy()
    Set_.drop('SaleStatus',axis=1,inplace=True) 
    Set_=np.asmatrix(Set_)
    InputSet_copy=np.delete(InputSet_copy,ColumnsToDelete,axis=1) 
    Set_=np.delete(Set_,ColumnsToDelete,axis=1) 
    del svc
    del selector
    
    #pca=PCA()
    #temp=pca.fit_transform(Set_)
    #PCA_Components_Set=pca.components_
    #InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCA_Components_Set))*np.transpose(np.asmatrix(InputSet_copy)))
    #del temp 
        
    svc=svm.SVC()
    parameters = {'kernel':('linear','rbf'), 'C':[100,200,300,400,500,600],'class_weight':('balanced',)}
    SVM = GridSearchCV(svc, parameters,cv=2)  
    SVM.fit(InputSet_copy,OutputSet_copy)
    a=pd.DataFrame(SVM.cv_results_)
    a.to_csv(directory+'Hyperparamter Data/SupportVectorMachine.csv')

    
    KernelData={}
    for i in a.index:
        if(a['param_kernel'][i] not in KernelData.keys()):
            KernelData[a['param_kernel'][i]]=[i]
        else:
            KernelData[a['param_kernel'][i]].append(i)  
    
    plt.figure()                
    for i in KernelData.keys():
        temp_C=[]
        temp_score=[]
        for j in KernelData[i]:
            temp_C.append(a['param_C'][j])
            temp_score.append(a['mean_test_score'][j])
        plt.plot(temp_C,temp_score,label=i)
    plt.xlabel('SVM Penalty "C" value')
    plt.ylabel('SVM cross validation score')     
    plt.title('Comparison of SVM paramters on score')
    plt.legend()
    #plt.show()
    del svc,SVM
    
    #Saving the model as pickle
    SVM=svm.SVC(kernel='rbf',C=100,class_weight='balanced')
    SVM.fit(InputSet_copy,OutputSet_copy)
    pickle_SVM=open(directory+'Models/SVM.pickle','wb')
    pickle.dump(SVM,pickle_SVM)
    pickle_SVM.close()
    


def RFClass(InputSet_copy,OutputSet_copy):
    '''
    Performs Random Forest Classification on Input. Function is divided into 5 parts.
    1st part: Finds the best features to perform classification on.
              Uses recursive feature elimination. Only those columns are retained
              in the Input Data which will be used for classification.
    2nd part: Use of principal component analysis. Decorrelates the input data
              by projecting input space into feature space where axes are 
              perpendicular to each other. Reduces overfitting to some extent.
    3rd part: Hyperparameter tuning using Grid Search. Results are stored in a
              separate .csv file.    
    4th part: Comparison of hyperparamters by effect on cross validation score 
    5th part: Saving model as pickle file                         
    '''
    global FeaturesToDelete
    InputSet_copy=np.asmatrix(InputSet_copy)
    OutputSet_copy=np.ravel(OutputSet_copy)
    rfc=RandomForestClassifier(n_estimators=11, bootstrap=True, criterion='entropy', max_depth=10, class_weight='balanced')
    selector = RFECV(rfc)
    selector = selector.fit(InputSet_copy,OutputSet_copy)
    SelectedFeatures=selector.support_ 
    ColumnsToDelete=[]
    for i in range(len(SelectedFeatures)):
        if(SelectedFeatures[i]==False):
            ColumnsToDelete.append(i)
    FeaturesToDelete['RFC']=ColumnsToDelete[:]         
    Set_=Set[:].copy()
    Set_.drop('SaleStatus',axis=1,inplace=True) 
    Set_=np.asmatrix(Set_)       
    InputSet_copy=np.delete(InputSet_copy,ColumnsToDelete,axis=1)  
    Set_=np.delete(Set_,ColumnsToDelete,axis=1)
    del rfc
    del selector
    
    pca=PCA()
    temp=pca.fit_transform(Set_)
    PCA_Components_Set=pca.components_
    InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCA_Components_Set))*np.transpose(np.asmatrix(InputSet_copy)))
    del temp
    
    parameters={'n_estimators':(10,20,30,40,50),'criterion':('gini','entropy'),'bootstrap':(True,),'class_weight':('balanced',) ,'max_depth':(5,10,15,20)}    
    rfc=RandomForestClassifier()
    RFC=GridSearchCV(rfc,parameters)
    RFC.fit(InputSet_copy,OutputSet_copy)
    a=pd.DataFrame(RFC.cv_results_)
    a.to_csv(directory+'Hyperparamter Data/RandomForest.csv')  
    
    
    KernelData={}
    for i in a.index:
        if(a['param_criterion'][i] not in KernelData.keys()):
            KernelData[a['param_criterion'][i]]=[i]
        else:
            KernelData[a['param_criterion'][i]].append(i)
     
    colors=['red','blue','green','yellow']  
    color_count=0                             
    for i in KernelData.keys():
        temp_MaxDepth=[]
        temp_Estimators=[]
        temp_score=[]  
        for j in KernelData[i]:
            temp_MaxDepth.append(a['param_max_depth'][j])
            temp_Estimators.append(a['param_n_estimators'][j])
            temp_score.append(a['mean_test_score'][j])
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(temp_MaxDepth, temp_Estimators, temp_score, label=i,c=colors[color_count]) 
        color_count+=1
        plt.xlabel('Depth of Tree')
        plt.ylabel('Number of Trees')
        plt.title('Comparison of RFC paramters on score')
        ax.legend()
        #plt.show()
    del rfc,RFC
    
       
    #Saving the PCA as pickle
    pickle_PCA_RFC=open(directory+'Models/PCA_RFC.pickle','wb')
    pickle.dump(PCA_Components_Set,pickle_PCA_RFC)
    pickle_PCA_RFC.close()  
    
    #Saving the model as pickle
    RFC=RandomForestClassifier(max_depth=10,n_estimators=41,criterion='entropy',class_weight='balanced')
    RFC.fit(InputSet_copy,OutputSet_copy)
    pickle_RFC=open(directory+'Models/RFC.pickle','wb')
    pickle.dump(RFC,pickle_RFC)
    pickle_RFC.close()            
                    

def NNClass(InputSet_copy,OutputSet_copy):
    '''
    Performs Neural Network on Input. 
    Function is divided into 4 parts.
    1st part: Maintains those features used by RFC. Only those columns are retained
              in the Input Data which will be used for classification.
    2nd part: Use of principal component analysis. Decorrelates the input data
              by projecting input space into feature space where axes are 
              perpendicular to each other. Reduces overfitting to some extent.
    3rd part: Hyperparameter tuning using Grid Search. Results are stored in a
              separate .csv file.    
    4th part: Comparison of hyperparamters by effect on cross validation score                          
    '''
    global FeaturesToDelete
    InputSet_copy=np.asmatrix(InputSet_copy)
    OutputSet_copy=np.ravel(OutputSet_copy)
    ColumnsToDelete=FeaturesToDelete['RFC'] 
    FeaturesToDelete['NN']=ColumnsToDelete[:]     
    Set_=Set[:].copy()
    Set_.drop('SaleStatus',axis=1,inplace=True) 
    Set_=np.asmatrix(Set_)   
    InputSet_copy=np.delete(InputSet_copy,ColumnsToDelete,axis=1)  
    Set_=np.delete(Set_,ColumnsToDelete,axis=1)       
    
    pca=PCA()
    temp=pca.fit_transform(Set_)
    PCA_Components_Set=pca.components_
    InputSet_copy=np.transpose(np.transpose(np.asmatrix(PCA_Components_Set))*np.transpose(np.asmatrix(InputSet_copy)))
    del temp        
    
    parameters={'hidden_layer_sizes':(10,20,30,40),'activation':('relu',),'solver':('lbfgs','adam'),'alpha':(0.01,0.1,1),'max_iter':(300,)}     #logistic avativation gives poorer result compared to ReLU
    nn=MLP()
    NN=GridSearchCV(nn,parameters)
    NN.fit(InputSet_copy,OutputSet_copy)
    a=pd.DataFrame(NN.cv_results_)
    a.to_csv(directory+'Hyperparamter Data/NeuralNetwork.csv')
    
    
    KernelData={}
    for i in a.index:
        if(a['param_solver'][i] not in KernelData.keys()):
            KernelData[a['param_solver'][i]]=[i]
        else:
            KernelData[a['param_solver'][i]].append(i)
     
    colors=['green','purple','red','blue',]  
    color_count=0                             
    for i in KernelData.keys():
        temp_alpha=[]
        temp_hidden_layer_sizes=[]
        temp_score=[]  
        for j in KernelData[i]:
            temp_alpha.append(a['param_alpha'][j])
            temp_hidden_layer_sizes.append(a['param_hidden_layer_sizes'][j])
            temp_score.append(a['mean_test_score'][j])
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(temp_alpha, temp_hidden_layer_sizes, temp_score, label=i,c=colors[color_count])
        color_count+=1
        plt.xlabel('Regularizaton Parameter')
        plt.ylabel('Neurons in hidden layer')
        plt.title('Comparison of NN paramters on score')
        ax.legend()
        #plt.show()
         
    #Saving the PCA as pickle
    pickle_PCA_MLP=open(directory+'Models/PCA_MLP.pickle','wb')
    pickle.dump(PCA_Components_Set,pickle_PCA_MLP)
    pickle_PCA_MLP.close()  
    
    #Saving the model as pickle
    mlp=MLP(hidden_layer_sizes=10,activation='relu',solver='lbfgs',alpha=3,max_iter=300)
    mlp.fit(InputSet_copy,OutputSet_copy)
    pickle_MLP=open(directory+'Models/MLP.pickle','wb')
    pickle.dump(mlp,pickle_MLP)
    pickle_MLP.close()    

#########################################################################PREPROCESSING###########################################################################################


Set=data_train.append(data_test)
Set=Set[Set.columns.sort_values()]

for i in Set.keys():
    if(type(Set[i][Set.index[1]])!=str):
        Set[i]=PreProcess(Set[i][:],i)
        
'''                                                                             
for i in Set.keys():                                                            #Converts nominal label to ordinal values        
    if(i!='SaleStatus' and  type(Set[i][Set.index[1]])==str):
        Set[i]=labelToOrdinal(Set[i][:],i)

for i in Set.keys():                                                           #Converts nominal labels to binary columnar values  
    if(i!='SaleStatus' and  type(Set[i][Set.index[1]])==str):
        labelToBinary(Set[i][:],i) 
'''        
for i in Set.keys():                                                           #Converts nominal labels to one-hot columnar values  
    if(i!='SaleStatus' and  type(Set[i][Set.index[1]])==str):
        labelToOneHot(Set[i][:],i)        
     
Set=pd.DataFrame(Set)                                                           #segregates the dataframe into input and output dataset
OutputSet=pd.DataFrame(Set['SaleStatus'])
OutputSet=OutputSet[0:1460].copy()
InputSet=Set.drop('SaleStatus',axis=1) 
InputSet=InputSet[0:1460].copy()  
InputSet.to_csv(directory+'ModifiedIP.csv')    


###############################################################################################################################################################################

print('Evaluating LDA 1/4...')
LDAClass(InputSet.copy(),OutputSet.copy())
print('Evaluating SVM 2/4...')
SVMClass(InputSet.copy(),OutputSet.copy())
print('Evaluating RFC 3/4...')
RFClass(InputSet.copy(),OutputSet.copy())
print('Evaluating Neural Network 4/4...')
NNClass(InputSet.copy(),OutputSet.copy())

#Saving the dataPreProcess as pickle
pickle_dataPreProcess=open(directory+'Models/dataPreProcess.pickle','wb')
pickle.dump(dataPreProcess,pickle_dataPreProcess)
pickle_dataPreProcess.close() 

#Saving the FeaturesToDelete as pickle
pickle_FeaturesToDelete=open(directory+'Models/FeaturesToDelete.pickle','wb')
pickle.dump(FeaturesToDelete,pickle_FeaturesToDelete)
pickle_FeaturesToDelete.close()       

#Saving the dataConversion as pickle
dataConversion['Mode']=Mode
dataConversion['Label']=Label
pickle_dC=open(directory+'Models/dataConversion.pickle','wb')
pickle.dump(dataConversion,pickle_dC)
pickle_dC.close() 

plt.show()      

