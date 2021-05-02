import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


algae= pd.read_csv('../data/algae.csv')
algae['Family']= 'Algae'
print(algae.head())
bats= pd.read_csv('../data/bats.csv')
bats['Family']= 'Bats'
print(bats.head())
cypra= pd.read_csv('../data/CypraeidaeTrain.csv')
cypra['Family']= 'Cypraeidae'
print(cypra.head())
dros= pd.read_csv('../data/Drosophila_train.csv')
dros['Family']= 'Drosophila'
print(dros.head())
fish= pd.read_csv('../data/fishes.csv')
fish['Family']= 'Fish'
print(fish.head())
fungi= pd.read_csv('../data/fungi.csv')
fungi['Family']= 'Fungi'
print(fungi.head())
inga= pd.read_csv('../data/IngaTrain.csv')
inga['Family']= 'Inga'
print(inga.head())

data= [algae, bats, cypra, dros, fish, fungi, inga]
combined= pd.concat(data)
combined.rename(columns={' Sequence':'Sequence'}, inplace=True)
#removing whitespace
combined['Properties']= combined['Properties'].apply(lambda x: x.strip())
#removing garbage data(noise)
combined= combined.loc[(combined['Properties']!='partial cds| |') & 
                       (combined['Properties']!='complete cds| |')]
#removing noise
combined['Sequence']= combined['Sequence'].apply(lambda x: x.replace('-', ''))
combined= combined.reset_index()
combined.drop('index', axis=1, inplace=True)
# =============================================================================
# print(combined['Sequence'].value_counts())
# print(combined['Properties'].describe())
# print(combined['Sequence'].describe())
# =============================================================================
combined['Properties']= combined['Properties'].str.split('|').str[1]
combined['Properties']= combined['Properties'].str.replace('_', ' ')
combined.rename(columns={'Properties':'Species'}, inplace=True)
#combined = combined.sample(frac=1).reset_index(drop=True) #shuffling the dataframe
print(combined.head())
print(combined.tail())
print(combined.shape)
print(combined.describe())
print(combined.info())
print(combined.columns)

#classifying only into species for now
y= combined['Species']

#label encoding the species name
label_y= y.copy()
label_encoder= LabelEncoder()
label_y= label_encoder.fit_transform(y)
print(label_y)

#one hot encoding the sequence
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

combined['Words'] = combined.apply(lambda x: getKmers(x['Sequence']), axis=1)
combined = combined.drop('Sequence', axis=1)
print(combined.head())

combined_texts = list(combined['Words'])
for item in range(len(combined_texts)):
    combined_texts[item] = ' '.join(combined_texts[item])
    
#Creating the Bag of Words model using CountVectorizer()
#This is equivalent to k-mer counting
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(combined_texts)
#print(X)
#Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=42)

#Multinomial Naive Bayes Classifier
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(y_pred)


print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
    f1 = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_pred))
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


test= input("Enter a DNA sequence to be classified: ")

