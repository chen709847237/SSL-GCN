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
The main script is ```local_run.py```.    
There are four input parameters:
1. ```-d``` The path to the data folder (with "/" or "\\" at the end)


```bash
python local_run.py -d <data_folder_path> -m <model_folder_path> -mt <model_type> -o <output_folder_path>
```
#### Example:
```bash
python local_run.py -mo ./model/ -da ./data/ -o ./
```

### 2. Final production model prediction  
The script is located in ```prediction``` folder
```bash
python prediction.py -t <tissue_type> -m <model_folder_path> -d <fasta_file_path> -o <output_folder_path>
```
where:  
```<tissue_type>``` could be selected from ```breast```, ```cervix```, ```colon```, ```lung```, ```prostate``` and ```skin```.   

#### Example:
```bash
python prediction.py -t breast -m ./model/ -d ./test_breast.fasta -o ./result/
```
**Note: The input peptide data must in the form of the following FASTA format.**
```bash
>AmphiArc1
KWVKKVHNWLRRWIKVFEALFG
>AmphiArc2
KIFKKFKTIIKKVWRIFGRF
>Gradient2
AWLKRIKKFLKALFWVWVW 
>AmphiArc3
AFRHSVKEELNYIRRRLERFPNRL
```
## References
1. We used ```iFeature``` to extract all peptide features. ([Github](https://github.com/Superzchen/iFeature/), [Paper](https://academic.oup.com/bioinformatics/article-abstract/34/14/2499/4924718))
