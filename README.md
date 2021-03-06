# SSL-GCN
## Compound Toxicity Prediction Based onSemi-supervised Learning and GraphConvolutional Neural Network

In this repository:         
We provide Python scripts to reproduce the experiments for conventional ML models, SL-GCN and SSL-GCN comparisons.

## Requirements 
* Anaconda 4.10.1
* Python 3.7.3
* Scikit-learn 0.23.2
* Pytorch 1.7.0 with CUDA 10.0
* Scipy 1.6.2
* Pandas 1.2.3 
* Numpy 1.19.2
* Openpyxl 3.0.7
* xgboost 1.3.3
* dgl 0.5.2
* dgllife 0.2.6
* joblib 1.0.1
* rdkit 2020.09.1

## Model and Data
Models and data used for reproducing experiments are available at: [[Data]](https://drive.google.com/file/d/1KSlG8LAdoINJwgbd0rN0L_5XYRK23znx/view?usp=sharing) [[Model]](https://drive.google.com/file/d/1xKz_zkinwA3BiqqAXOjHAYtVilWp-Zlz/view?usp=sharing)

## Reproducing Experiments
### 1. Download model and data
Unzip the downloaded ```data.7z``` and ```model.7z``` files, place the ```data``` folder and ```model``` folder in the same folder as the scripts.    

![image](https://github.com/chen709847237/SSL-GCN/raw/main/img/data_sample.png)     

### 2.  Run the script  
The main script is ```local_run.py```. There are four input parameters for this script:      
```bash
python local_run.py -d <data_folder_path> -m <model_folder_path> -mt <model_type> -o <output_folder_path>
```
```-d```：The path to the data folder (with "/" or "\\" at the end).         
```-m```：The path to the model folder (with "/" or "\\" at the end).           
```-mt```：Define the type of model, ```cm``` - conventional ML models, ```sl``` - SL-GCN models, ```ssl``` - SSL-GCN models.                  
```-o```：The path to an empty output folder where the experiment results will be stored (with "/" or "\\" at the end).     
#### Example:
```bash
python local_run.py -d ./data/ -m ./model/ -mt cm -o ./cm_output_result/
```
#### Note:
Running time for SL-GCN models is approx ```3 min```.      
Running time for SSL-GCN models is approx ```13 min```.      
Running time for CM models is approx ```32 min```.      

### 3. Result  
After running, there should be two types of files in the output folder.           
As the following figure shows, the result files of SL-GCN models.              

![image](https://github.com/chen709847237/SSL-GCN/raw/main/img/result_sample.png)     

File in the RED box contains the average test performance (average AUC scores) of SL-GCN models on the 12 prediction tasks in 5 repeated experiments.        
Files in the GREEN box contain the detailed AUC scores of SL-GCN models during 5 repeated experiments on the 12 prediction tasks.         

## References
1. In this study, the Tox21 dataset from MoleculeNet ([Website](http://moleculenet.ai/), [Github](https://github.com/deepchem/deepchem), [Paper](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a)) is used as the labeled data.
2. The GCN model is implemented using DGL 0.5.6 ([Github](https://github.com/dmlc/dgl/)) and its supplementary package DGL-LifeSci 0.2.6 ([Github](https://github.com/awslabs/dgl-lifesci/))
