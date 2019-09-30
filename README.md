# MIC: Mining Interclass Characteristics for Improved Deep Metric Learning
---
#### ___Authors___:
* Karsten Roth (karsten.rh1@gmail.com)
* Biagio Brattoli (biagio.brattoli@gmail.com)
* Björn Ommer

___Primary Contact___: Karsten Roth

---
This repository contains the code to run the pipeline proposed in our ICCV 2019 paper _Mining Interclass Characteristics for Improved Deep Metric Learning_ (https://arxiv.org/abs/1909.11574). The results using this pipeline for ProxyNCA and Triplet with Semihard Sampling are better than noted in the paper due to an improved implementation of the baseline methods.

**Note:** Baseline implementations can be found at https://github.com/Confusezius/Deep-Metric-Learning-Baselines.

---
### Requirements
Our method was tested around
* Python Version 3.6.6+
* PyTorch Version 1.0.1+
* Faiss(-gpu) 1.5.1 (GPU support optional)
* Scikit Image 0.14.2
* Scikit Learn 0.20.3
* Scipy 1.2.1

To run with standard batch sizes, at least 11 GB of VRAM is required (e.g. 1080Ti, Titan X).

---

### How To Use
For a quick start for standard Deep Metric Learning datasets:

* [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200.html)
* [CARS196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/)
* [In-Shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
* [PKU Vehicle-ID](https://www.pkuml.org/resources/pku-vds.html)


simply run the sample setups given in `Result_Runs.sh`. These give similar values (*assuming the underlying setup to be the same*) as those reported in the paper. Minor differences are due to choice of seeds and underlying setups.  

---

### Specifics
The main script is `main.py`. Running it with default flags will provide a Metric Learning Run with Interclass Mining on CUB200-2011 using Resnet50, Marginloss and Distance-weighted Sampling. For all tweakable parameters and their purpose, please refer to the help-strings in the `main.py`-ArgumentParser. Most should be fairly self-explanatory. Again, good default setups can be found in `Result_Runs.sh`.   

__NOTE__ regarding ProxyNCA for __Online Products__, __PKU Vehicle ID__ and __In-Shop Clothes__: Due to the high number of classes, the number of proxies required is too high for useful training (>10000 proxies).


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
| checkpoint.pth.tar     -> Contains network state-dict.
| hypa.pkl               -> Contains all network parameters as pickle.
|                           Can be used directly to recreate the network.
| log_train_Class.csv    -> Logged training data as CSV.                      
| log_val_Class.csv      -> Logged test metrics as CSV.                    
| Parameter_Info.txt     -> All Parameters stored as readable text-file.
| InfoPlot_Class.svg     -> Graphical summary of training/testing metrics progression.
| Curr_Summary_Class.txt -> Summary of training (best metrics...).                      
| sample_recoveries.png  -> Sample recoveries for best validation weights.
|                           Acts as a sanity test.
```

---

## Citing Our Paper
If you use this repository or wish to cite our results, please use (https://arxiv.org/abs/1909.11574)
```
@conference{roth2019mic,
  title={MIC: Mining Interclass Characteristics for Improved Metric Learning},
  author={Roth, Karsten, and Brattoli, Biagio, and Ommer, Bj\"orn},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
