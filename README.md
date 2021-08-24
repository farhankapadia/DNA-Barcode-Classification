# DNA-Barcode-Classification
 
**Classification** of an **unknown** specimen to its respective **Taxonomy** rank by analyzing its **DNA Barcode**.  
  
### Input  
DNA Barcode as a String

### Output  
Efficacious classification analysis of the given specimen to its respective taxonomy rank with the help of Supervised Machine Learning algorithms.  

## Screenshots
|||
|--|--|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/1.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/2.png)|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/4.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/3.png)|

## Dataset Description
|No.| Dataset| #seq.| seq.length| #species| Gene region(s)|
|--|--|--|--|--|--|
|1| Bats |826| 659| 82| COI|
|2 |Fish |626 |419 |82 |COI|
|3 |Birds |1936| 703| 575| COI|
|4| Amphibian |357 |669 |30 |COI|
|5 |Plants (Inga)| 913| 1,838| 56| tmTD, ITS| 
|6 |Sea snail (Cypraeidae) |2,008| 614|211| COI| 
|7 |Fruit Flies (Drosophila) |615 |663| 19| COI|
|8 |Butterfly| 1235| 658| 174| COI| 
|9 |Fungi |50 |510 |8 |ITS| 
|10 |Algae| 26| 1,128| 5 |rbcL|   
  

### Total number of Species per Family after cleaning
|Family| Total number of Species| 
|--|--|
|Algae |25 |
|Amphibians| 274 |
|Bats |839 |
|Birds |1396 |
|Butterfly| 926| 
|Fish |625 |
|Fruit Flies| 498 |
|Fungi |48 |
|Plants |785| 
|Sea Snail |1655|

###  Unique labels of Species per Family after cleaning
|Family| Unique Labels of Species| 
|--|--|
|Algae |5 |
|Amphibians| 29 |
|Bats |96 |
|Birds |574 |
|Butterfly| 174| 
|Fish |82 |
|Fruit Flies| 19 |
|Fungi |8 |
|Plants |61| 
|Sea Snail |211|

|||
|--|--|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/totalSpeciesInEachFamily.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/uniqueSpeciesInEachFamily.png)|

## Performance Metrics
The 4 algorithms are tested over the 4 performance metrics to predict the ‘Species’ and ‘Family’. Total time taken for each algorithm to predict both the labels is considered as the runtime of that algorithm. Following Tables and graphs below indicate the results and further evaluation is comprehended on it’s basis. 

### Species Classification
||Naïve Bayes| SVM| Random Forest| kNN|
|--|--|--|--|--|
|Accuracy| 0.890| 0.878| 0.901 |0.871|
|Precision| 0.965| 0.969 |0.967| 0.964| 
|Recall| 0.978| 0.971| 0.985 |0.966| 
|F1 |0.966| 0.964| 0.970| 0.959| 
### Family Classification
||Naïve Bayes| SVM| Random Forest| kNN|
|--|--|--|--|--|
|Accuracy| 1.000|1.000|1.000|0.978|
|Precision|1.000|1.000|1.000| 0.985|
|Recall|1.000|1.000|1.000|0.978|
|F1|1.000|1.000|1.000|0.980|

|||
|--|--|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/accuracySpecies.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/accuracyFamily.png)|

### Runtime 
The specifications of the machine on which the implementations were performed and measured 
are:   
* **Model** - MacBook Pro     
* **Processor** - Intel R Core i5   
* **Installed RAM** - 8 GB   
* **System type** - 64-bit OS, x64 based processor   
* **OS** - Windows 10 Pro   

||Naïve Bayes| SVM| Random Forest| kNN|
|--|--|--|--|--|
|Runtime (in seconds)| 9.288| 202.471| 86.966| 3.609|
