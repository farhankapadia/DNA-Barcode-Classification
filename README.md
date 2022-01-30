# DNA-Barcode-Classification
 
 *B.E. Final Year Research Project 2020-2021*
 
 ## Description
**Classification** of an **unknown** specimen to its respective **Taxonomy** rank by analyzing its **DNA Barcode**.  
  
### Input  
1. DNA Barcode as a raw FASTA sequence (string).  

2. Choice of **Supervised Machine Learning** Model to use for prediction:
   * *Naïve Bayes*
   * *Support vector machine*
   * *Random forest*
   * *k-nearest neighbors algorithm*

### Output  
1. Classification analysis of the given specimen to its respective taxonomy rank:
   * *Dataset*
   * *Family*
   * *Species*

2. Image of the Dataset it belongs to.

3. DNA Barcode string converted to an image form where single character corresponds to a strip of colour code as:
   *  A --> Green
   *  C --> Blue
   *  T --> Red
   *  G --> Black

4. Performance Metrics of the chosen ML model for Label Prediction (Species and Family).   

## Technology
* Python 
   * **NumPy**
   * **pandas**
   * **scikit-learn**
   * **Matplotlib**
   * **Seaborn**
   * **fasta2csv** : converting FASTA format to CSV
   * **Tkinter** : GUI
   * **Pillow**
   * **os**
   * **time**


## Dataset
### Collection
The **Train** and **Test** Datasets in FASTA format can be found in the [data](https://github.com/farhankapadia/DNA-Barcode-Classification/tree/master/data) folder.  

The data which we used was cleaned, merged and manipulated to match our needs. This was extracted from Empirical Datasets of research papers. Links for the Datasets: 
* https://github.com/zhangab2008/BarcodingR/blob/master/Appendix_S2_empiricalDatasets.zip 
*  http://dmb.iasi.cnr.it/supbarcodes.php 

### Description 
We are using datasets from 10 different organisms with multiple numbers of species within them. Table below gives the summary of the dataset.  

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
  

### Data after Cleaning
Raw FASTA files were cleaned by removing garbage values and redundant characters. They were then converted to CSV format for training the models.  

|Family| #sequences| #species |
|--|--|--|
|Algae |25 |5|
|Amphibians| 274 | 29|
|Bats |839 |96
|Birds |1396 | 574|
|Butterfly| 926| 174 |
|Fish |625 | 82 |
|Fruit Flies| 498 | 19 |
|Fungi |48 | 8|
|Plants |785| 61|
|Sea Snail |1655| 211 |


|#sequences|#species|
|--|--|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/totalSpeciesInEachFamily.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/images/uniqueSpeciesInEachFamily.png)|

## Performance Metrics
The 4 algorithms : *Naïve Bayes*, *SVM*, *Random Forest* and *kNN* are tested over the 4 performance metrics : *Accuracy*, *Precison*, *Recall* and *F1-score* to predict the **Species** and **Family**.  
 Total time taken for each algorithm to predict both the labels is considered as the *Runtime* of that algorithm.  
  
  Following Tables and graphs below indicate the results and further evaluation is comprehended on it’s basis. 

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

|Species Classification|Family Classification|
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

Thus, **Naïve Bayes** seems to be the winner amongst the 4 algorithms for it's nearly best *Perfromance metrics* along with second best *Runtime*.

## Screenshots
|||
|--|--|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/1.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/2.png)|
|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/4.png)|![alt text](https://github.com/farhankapadia/DNA-Barcode-Classification/blob/master/screenshots/3.png)|

## Setup
1. Clone this repository or download zip.  
2. Open this repository on terminal. Navigate to [src](https://github.com/farhankapadia/DNA-Barcode-Classification/tree/master/src) folder by typing ```cd src```.

3. Type (if mentioned above **python modules** are not installed)  

   ```
   pip install pandas sklearn matplotlib seaborn
   ``` 
   
4. To run the project,
   ```
    python main.py
   ```
5. All set.
