# MIC: Mining Interclass Characteristics to improve Deep Metric Learning
---
#### ___Authors___:
* Karsten Roth (karsten.roth@stud.uni-heidelberg.de)
* Biagio Brattoli (biagio.brattoli@gmail.com)
* Björn Ommer ()

---
This repository contains the code to run the pipeline proposed in our ICCV 2019 paper _Mining Interclass Characteristics for Improved Deep Metric Learning_ (arxiv link).

**Note:** Baseline values referenced were created around (github link)

---
### Requirements
Our method was tested around
* Python Version 3.6.6+
* PyTorch Version 1.0.1+
* Faiss (GPU support optional)
* Scikit Image 0.14.2
* Scikit Learn 0.20.3
* Scipy 1.2.1

To run with standard batch sizes, at least 11 GB of VRAM is required (e.g. 1080Ti, Titan X).

---

### How To Use
For a quick start for standard Deep Metric Learning datasets:

* CUB200-2011 ()
* CARS196 ()
* Stanford Online Products ()
* In-Shop Clothes ()
* PKU Vehicle-ID ()


simply run the sample setups given in `Result_Runs.sh`. These give the values reported in the paper.  

---

### Specifics
The main script is `main.py`. Running it with default flags will provide a Metric Learning Run with Interclass Mining on CUB200-2011 using Resnet50, Marginloss and Distance-weighted Sampling. For all tweakable parameters and their purpose, please refer to the help-strings in the `main.py`-ArgumentParser. Most should be fairly self-explanatory.  
__NOTE:__
ProxyNCA for __Online Products__, __PKU Vehicle ID__ and __In-Shop Clothes__: Due to the high number of classes, the number of proxies required is too high for useful training (>10000 proxies).


---

### Repo Structure
```
Repository
│   README.md
|
|   ### Main Scripts
|   main.py     (main training script)
|   losses.py   (collection of loss and sampling impl.)
│   datasets.py (dataloaders for all datasets)
│   
│   ### Utility scripts
|   auxiliaries.py  (set of utilities)
|   evaluate.py     (set of evaluation functions)
│   
│   ### Network Scripts
|   netlib.py       (contains impl. for ResNet50 and network utils)
|   googlenet.py    (contains impl. for GoogLeNet)
│   
└───Training Results (generated during Training)
|    │   e.g. cub200/Training_Run_Name
|    │   e.g. cars196/Training_Run_Name
│   
└───Datasets (should be added, if one does not want to set paths)
|    │   cub200, cars196 ...
```

---


### Required Dataset Structures
__CUB200-2011__
```
cub200
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

__CARS196__
```
cars196
└───images
|    └───Acura Integra Type R 2001
|           │   00128.jpg
|           │   ...
|    ...
```

__Online Products__
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

__In-Shop Clothes__
```
in-shop
└───img
|    └───MEN
|         └───Denim
|               └───id_00000080
|                       │   01_1_front.jpg
|                       │   ...
|    ...
└───Eval
|    │   list_eval_partition.txt
```


__PKU Vehicle ID__
```
vehicle_id
└───image
|     │   <img>.jpg
|     |   ...
└───train_test_split
|     |   test_list_800.txt
|     |   ...
```

---

### Stored Data:
By default, the following files are saved:
```
Name_of_Training_Run
|  checkpoint.pth.tar   -> Contains network state-dict.
|  hypa.pkl             -> Contains all network parameters as pickle.
|                          Can be used directly to recreate the network.
| log_train_Class.csv    -> Logged training data as CSV.                      
| log_val_Class.csv      -> Logged test metrics as CSV.                    
| Parameter_Info.txt    -> All Parameters stored as readable text-file.
| InfoPlot_Class.svg     -> Graphical summary of training/testing metrics progression.
| sample_recoveries.png -> Sample recoveries for best validation weights.
|                          Acts as a sanity test.
```

---

## Expected Results
These results are expected for each dataset when running our method as provided in `Result_Runs.sh`.
By default, our reported values use Marginloss with Distance-weighted Sampling as basis.
_Note also that there is a not insignificant dependency on the used seed._


__CUB200-2011__

   NMI  |  F1  | R@1  | R@2   | R@4   | R@8
--------|------|----- |-----  |-----  |----
 68.2   | 38.7 | 63.4 | 74.9  |  86.0 |  90.4    

__CARS196__

   NMI  |  F1  | R@1  | R@2   | R@4   | R@8
--------|------|----- |-----  |-----  |----
 68.2   | 38.7 | 63.4 | 74.9  |  86.0 |  90.4    

__STANFORD ONLINE PRODUCTS__

   NMI  |  F1  | R@1  | R@10   | R@100   | R@1000
--------|------|----- |-----  |-----  |----
 68.2   | 38.7 | 63.4 | 74.9  |  86.0 |  90.4    

__IN-SHOP CLOTHES__

   NMI  |  F1  | R@1  | R@10   | R@20   | R@30   | R@50
--------|------|----- |-----  |-----  |----   |----
 68.2   | 38.7 | 63.4 | 74.9  |  86.0 |  90.4 |  90.4    

__PKU VEHICLE ID__

   NMI  |  F1  | Small R@1  | Small R@5 | Big R@1 | Big R@5   
--------|------|----- |----- |----- |-----
 68.2   | 38.7 | 63.4 | 74.9 | 77.7 | 77.7
