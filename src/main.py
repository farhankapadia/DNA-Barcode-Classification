import pandas as pd
import re

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
inga['Family']= 'IngaTrain'
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
print(combined.head())
print(combined.tail())
print(combined.shape)
print(combined.describe())
print(combined.info())
print(combined.columns)