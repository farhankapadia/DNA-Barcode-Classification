import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns


algae= pd.read_csv('../data/algae.csv')
algae['Family']= 'Algae'
print(algae.head())
bats= pd.read_csv('../data/bats.csv')
bats['Family']= 'Bats'
print(bats.head())
cypra= pd.read_csv('../data/CypraeidaeTrain.csv')
cypra['Family']= 'Sea Snail'
print(cypra.head())
dros= pd.read_csv('../data/Drosophila_train.csv')
dros['Family']= 'Fruit Flies'
print(dros.head())
fish= pd.read_csv('../data/fishes.csv')
fish['Family']= 'Fish'
fish= fish.reset_index()
fish.drop('Properties', axis=1, inplace=True)
fish.rename(columns={'index':'Properties'}, inplace=True)
print(fish.columns)
print(fish.head())
fungi= pd.read_csv('../data/fungi.csv')
fungi['Family']= 'Fungi'
print(fungi.head())
inga= pd.read_csv('../data/IngaTrain.csv')
inga['Family']= 'Plants'
print(inga.head())
amph= pd.read_csv('../data/Amphibian_Train.csv')
amph['Family']= 'Amphibians'
print(amph.head())
birds= pd.read_csv('../data/bird_Train.csv')
birds['Family']= 'Birds'
print(birds.head())
butterfly= pd.read_csv('../data/butterfly_Train.csv')
butterfly['Family']= 'Butterfly'
print(butterfly.head())


data1= [amph, birds, butterfly]
data1= pd.concat(data1)
data1['Properties']= data1['Properties'].str.split('unknownFamily').str[1]
data1['Properties']= data1['Properties'].str.replace('_', ' ')
data1['Properties']= data1['Properties'].apply(lambda x: x.strip())
data1.rename(columns={' Sequence':'Sequence'}, inplace=True)
data1['Sequence']= data1['Sequence'].apply(lambda x: x.replace('-', ''))
data1= data1.reset_index()
data1.drop('index', axis=1, inplace=True)


data= [algae, bats, cypra, dros, fish, fungi, inga]
data= pd.concat(data)
data.rename(columns={' Sequence':'Sequence'}, inplace=True)
data['Properties']= data['Properties'].str.split('|').str[1]
print(data.loc[data['Family']=='Fish'])
#removing whitespace
data['Properties']= data['Properties'].apply(lambda x: x.strip())
#removing garbage data(noise)

data= data.loc[(data['Properties']!='partial cds| |') & 
                       (data['Properties']!='complete cds| |')]

#removing noise
data['Sequence']= data['Sequence'].apply(lambda x: x.replace('-', ''))
data= data.reset_index()
data.drop('index', axis=1, inplace=True)
# =============================================================================
# print(data['Sequence'].value_counts())
# print(data['Properties'].describe())
# print(data['Sequence'].describe())
# =============================================================================
data['Properties']= data['Properties'].str.replace('_', ' ')
combined= [data, data1]
combined= pd.concat(combined)
combined= combined.reset_index()
combined.drop('index', axis=1, inplace=True)
combined.rename(columns={'Properties':'Species'}, inplace=True)

print(combined.head())
print(combined.tail())
print(combined.shape)
print(combined.describe())
print(combined.info())
print(combined.columns)

#classifying into species and family
y= combined['Species']
y1= combined['Family']

# =============================================================================
# #one hot encoding the sequence
# def string_to_array(my_string):
#     my_string = my_string.lower()
#     my_string = re.sub('[^acgt]', 'z', my_string)
#     my_array = np.array(list(my_string))
#     return my_array
# =============================================================================

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

combined['Words'] = combined.apply(lambda x: getKmers(x['Sequence']), axis=1)
combined = combined.drop('Sequence', axis=1)
print(combined.head())
print(combined.columns)

combined_texts = list(combined['Words'])
    
test_sequence= input("Enter a DNA sequence to be classified: ")
test_sequence= test_sequence.replace('-', '')
test_sequence= getKmers(test_sequence)
new_test_sequence= ' '.join(test_sequence)
#combined_texts.append(test_sequence)
for item in range(len(combined_texts)):
    combined_texts[item] = ' '.join(combined_texts[item])
    
#label encoding the species name
label_y= y.copy()
label_encoder= LabelEncoder()
label_y= label_encoder.fit_transform(y)

#Creating the Bag of Words model using CountVectorizer()
#This is equivalent to k-mer counting
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(combined_texts)

#Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=42)

X1_train, X1_test, y1_train, y1_test = train_test_split(X, 
                                                    y1, 
                                                    test_size = 0.2, 
                                                    random_state=42)

#Multinomial Naive Bayes Classifier
def MultiNB():
    model= MultinomialNB(alpha=0.01)
    return model

#SVM Classifier
def SVM():
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    return model

#Random Forest Classifier
def RandomForest():
    model= RandomForestClassifier(n_estimators=100, criterion='gini')
    return model

#KNN
def kNN():
    model= KNeighborsClassifier(n_neighbors=2, weights='distance')
    return model


def chooseModel(classifier):
    global total_time, clf, y_pred, y1_pred
    start_time= time.time()
    
    #classifier = kNN() #just change the method name to call a different model
    clf= classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    
    new_x= cv.transform([new_test_sequence])
    new_y= classifier.predict(new_x)
    print("The predicted species is: ", new_y[0]) #"new_y[0] is the variable that has the species name stored
    
    classifier.fit(X1_train, y1_train)
    new_x1= cv.transform([new_test_sequence])
    new_y1= classifier.predict(new_x1)
    print("The predicted family is: ", new_y1[0]) #"new_y1[0]" is the family name
    
    y1_pred= classifier.predict(X1_test)
    
    end_time= time.time()
    total_time= end_time - start_time
    
#metrics for species classification
def metrics(y_test, y_pred):
    global accuracy_s, precision_s, recall_s, f1_s
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        f1 = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        return accuracy, precision, recall, f1
    accuracy_s, precision_s, recall_s, f1_s = get_metrics(y_test, y_pred)
    print("The metrics for species classification are: ")
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy_s, precision_s, recall_s, f1_s))
    

#metrics for family classification
def metricsFamily(y_test, y_pred):
    global accuracy_f, precision_f, recall_f, f1_f
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        f1 = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        return accuracy, precision, recall, f1
    accuracy_f, precision_f, recall_f, f1_f = get_metrics(y_test, y_pred)
    print("The metrics for family classification are: ")
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy_f, precision_f, recall_f, f1_f))


def getParams():
    parameters= {'n_neighbors': [1, 2, 3]}
    gs_clf= GridSearchCV(clf, parameters)
    gs_clf= gs_clf.fit(X_train, y_train)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

accuracyS= []
precisionS= []
recallS= []
f1S= []

accuracyF= []
precisionF= []
recallF= []
f1F= []

def printMetrics():
    metrics(y_test, y_pred)
    metricsFamily(y1_test, y1_pred)
    print("The total runtime of the model is: ", total_time)


def appendSMetrics():
    accuracyS.append(accuracy_s)
    precisionS.append(precision_s)
    recallS.append(recall_s)
    f1S.append(f1_s)
    
def appendFMetrics():
    accuracyF.append(accuracy_f)
    precisionF.append(precision_f)
    recallF.append(recall_f)
    f1F.append(f1_f)
   
def callFuncs():
    printMetrics()
    appendSMetrics()
    appendFMetrics()


#getParams() #do not uncomment. Processing time is ridiculous

#calling all models one by one

#Naive Baye's first
chooseModel(MultiNB())
callFuncs()


# =============================================================================
# #the below commented cell is not needed as we will only use 1 algo
# #for our project
# #this can be uncommented anytime to run all 4 algos and get graphs
# #SVM next
# chooseModel(SVM())
# callFuncs()
# #Random Forest next
# chooseModel(RandomForest())
# callFuncs()
# #kNN last
# chooseModel(kNN())
# callFuncs()
# 
# #barplot for comparing accuracies of all 4 algos for species
# algos= ["Naive Baye's", "SVM", "Random Forest", "kNN"]
# accuracySper= [i * 100 for i in accuracyS] 
# plt.figure(figsize=(10,6))
# plt.title('Accuracy of the algorithms for species')
# plt.xlabel('Algorithms')
# plt.ylabel('Accuracy')
# sns.barplot(x=algos, y=accuracySper)
# 
# #barplot for comparing accuracies of all 4 algos for family
# accuracyFper= [i * 100 for i in accuracyF]
# plt.figure(figsize=(10,6))
# plt.title('Accuracy of the algorithms for family')
# plt.xlabel('Algorithms')
# plt.ylabel('Accuracy')
# sns.barplot(x=algos, y=accuracyFper)
# 
# #barplot for comparing precision of all 4 algos for species
# precisionSper= [i * 100 for i in precisionS]
# plt.figure(figsize=(10,6))
# plt.title('Precision of the algorithms for species')
# plt.xlabel('Algorithms')
# plt.ylabel('Precision')
# sns.barplot(x=algos, y=precisionSper)
# 
# #barplot for comparing precision of all 4 algos for family
# precisionFper= [i * 100 for i in precisionF]
# plt.figure(figsize=(10,6))
# plt.title('Precision of the algorithms for family')
# plt.xlabel('Algorithms')
# plt.ylabel('Precision')
# sns.barplot(x=algos, y=precisionFper)
# 
# #barplot for comparing recall score of all 4 algos for species
# recallSper= [i * 100 for i in recallS]
# plt.figure(figsize=(10,6))
# plt.title('Recall score of the algorithms for species')
# plt.xlabel('Algorithms')
# plt.ylabel('Recall score')
# sns.barplot(x=algos, y=recallSper)
# 
# #barplot for comparing recall score of all 4 algos for family
# recallFper= [i * 100 for i in recallF]
# plt.figure(figsize=(10,6))
# plt.title('Recall score of the algorithms for family')
# plt.xlabel('Algorithms')
# plt.ylabel('Recall score')
# sns.barplot(x=algos, y=recallFper)
# 
# #barplot for comparing F1 score of all 4 algos for species
# f1Sper= [i * 100 for i in f1S]
# plt.figure(figsize=(10,6))
# plt.title('F1 score of the algorithms for species')
# plt.xlabel('Algorithms')
# plt.ylabel('F1 score')
# sns.barplot(x=algos, y=f1Sper)
# 
# #barplot for comparing F1 score of all 4 algos for family
# f1Fper= [i * 100 for i in f1F]
# plt.figure(figsize=(10,6))
# plt.title('F1 score of the algorithms for family')
# plt.xlabel('Algorithms')
# plt.ylabel('F1 score')
# sns.barplot(x=algos, y=f1Fper)
# =============================================================================


#total number of species in each family
families= ['Algae', 'Amphibians', 'Bats', 'Birds', 'Butterfly',
           'Fish', 'Fruit Flies', 'Fungi', 'Plants', 'Sea Snail']
each_family= combined.groupby('Family').count()
print(combined.groupby('Family').count())
plt.figure(figsize=(10,6))
plt.title("No. of species in each Family")
plt.xlabel('Family')
plt.ylabel('No. of species')
sns.barplot(x=families, y=each_family['Species'])

#total number of unique species in each family
each_family= combined.groupby('Family').Species.nunique()
print(combined.groupby('Family').Species.nunique())
plt.figure(figsize=(10,6))
plt.title("No. of species in each Family")
plt.xlabel('Family')
plt.ylabel('No. of species')
sns.barplot(x=families, y=each_family.values)


species_fam= combined.groupby('Family').Species.value_counts()
print(species_fam)


#plots are for species in each family
# =============================================================================
# #barplot for no. of each unique species for algae
# plt.figure(figsize=(10,6))
# plt.title("No. of each unique species in Algae")
# plt.xlabel('No. of species')
# plt.ylabel('Species')
# sns.barplot(x=species_fam.loc['Algae'].values, y=species_fam.loc['Algae'].index, orient='h')
# 
# #barplot for no. of each unique species for amphibians
# plt.figure(figsize=(15,6))
# plt.title("No. of each unique species in Amphibians")
# plt.xlabel('No. of species')
# plt.ylabel('Speceis')
# sns.barplot(x=species_fam.loc['Amphibians'].values, y=species_fam.loc['Amphibians'].index, orient='h')
# 
# #barplot for no. of each unique species for bats
# plt.figure(figsize=(15,20))
# plt.title("No. of each unique species in Bats")
# plt.xlabel('No. of species')
# plt.ylabel('Species')
# sns.barplot(x=species_fam.loc['Bats'].values, y=species_fam.loc['Bats'].index, orient='h')
# 
# #barplot for no. of each unique species for Birds
# plt.figure(figsize=(20,100))
# plt.title("No. of each unique species in Birds")
# plt.xlabel('No. of species')
# plt.ylabel('Species')
# sns.barplot(x=species_fam.loc['Birds'].values, y=species_fam.loc['Birds'].index, orient='h')
# 
# #barplot for no. of each unique species for bats
# plt.figure(figsize=(15,40))
# plt.title("No. of each unique species in Butterfly")
# plt.xlabel('No. of species')
# plt.ylabel('Species')
# sns.barplot(x=species_fam.loc['Butterfly'].values, y=species_fam.loc['Butterfly'].index, orient='h')
# 
# #barplot for no. of each unique species for Fish
# plt.figure(figsize=(15,30))
# plt.title("No. of each unique species in Fish")
# plt.xlabel('No. of species')
# plt.ylabel('Species')
# sns.barplot(x=species_fam.loc['Fish'].values, y=species_fam.loc['Fish'].index, orient='h')
# 
# #barplot for no. of each unique species for Fruit Flies
# plt.figure(figsize=(15,6))
# plt.title("No. of each unique species in Fruit Flies")
# plt.xlabel('No. of species')
# plt.ylabel('Speceis')
# sns.barplot(x=species_fam.loc['Fruit Flies'].values, y=species_fam.loc['Fruit Flies'].index, orient='h')
# 
# #barplot for no. of each unique species for fungi
# plt.figure(figsize=(10,6))
# plt.title("No. of each unique species in Fungi")
# plt.xlabel('No. of species')
# plt.ylabel('Speceis')
# sns.barplot(x=species_fam.loc['Fungi'].values, y=species_fam.loc['Fungi'].index, orient='h')
# 
# #barplot for no. of each unique species for plants
# plt.figure(figsize=(15,30))
# plt.title("No. of each unique species in Plants")
# plt.xlabel('No. of species')
# plt.ylabel('Speceis')
# sns.barplot(x=species_fam.loc['Plants'].values, y=species_fam.loc['Plants'].index, orient='h')
# 
# #barplot for no. of each unique species for sea snail
# plt.figure(figsize=(15,75))
# plt.title("No. of each unique species in Sea Snail")
# plt.xlabel('No. of species')
# plt.ylabel('Speceis')
# sns.barplot(x=species_fam.loc['Sea Snail'].values, y=species_fam.loc['Sea Snail'].index, orient='h')
# 
# =============================================================================




