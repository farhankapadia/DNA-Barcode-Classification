import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


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
combined.rename(columns={'Properties':'Species'}, inplace=True)
#combined = combined.sample(frac=1).reset_index(drop=True) #shuffling the dataframe
print(combined.head())
print(combined.tail())
print(combined.shape)
print(combined.describe())
print(combined.info())
print(combined.columns)

#classifying only into species for now
X= combined['Sequence']
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

def one_hot_encoder(my_array):
    label_encoder.fit(np.array(['a','c','g','t','z']))
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, n_values=5)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded

label_X= X.copy()