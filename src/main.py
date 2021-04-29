import pandas as pd

algae= pd.read_csv('../data/algae.csv')
print(algae.head())
bats= pd.read_csv('../data/bats.csv')
print(bats.head())
cypra= pd.read_csv('../data/CypraeidaeTrain.csv')
print(cypra.head())
dros= pd.read_csv('../data/Drosophila_train.csv')
print(dros.head())
fish= pd.read_csv('../data/fishes.csv')
print(fish.head())
fungi= pd.read_csv('../data/fungi.csv')
print(fungi.head())
inga= pd.read_csv('../data/IngaTrain.csv')
print(inga.head())

data= [algae, bats, cypra, dros, fish, fungi, inga]
combined= pd.concat(data)
print(combined.head())
print(combined.tail())