import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


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
data['Properties']= data['Properties'].str.split('|').str[1]
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
    model= RandomForestClassifier()
    return model


classifier = MultiNB() #just change the method name to call a different model
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

#metrics for species classification
def metrics(y_test, y_pred):
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        f1 = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        return accuracy, precision, recall, f1
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("The metrics for species classification are: ")
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    

#metrics for family classification
def metricsFamily(y_test, y_pred):
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        f1 = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
        return accuracy, precision, recall, f1
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("The metrics for family classification are: ")
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


def getParams():
    parameters= {'alpha' : [1e-1, 1e-2, 1e-3, 1e-4]}
    gs_clf= GridSearchCV(clf, parameters)
    gs_clf= gs_clf.fit(X_train, y_train)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

metrics(y_test, y_pred)
metricsFamily(y1_test, y1_pred)
getParams()



