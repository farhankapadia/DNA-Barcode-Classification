import pandas as pd

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
print(combined.head())
print(combined.tail())