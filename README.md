# Gender Artifacts in Visual Datasets
### [Project Page](https://princetonvisualai.github.io/gender-artifacts/) | [Paper](https://arxiv.org/abs/2206.09191)

This repo provides the code for the paper "Gender Artifacts in Visual Datasets."

```
  @article{meister2022artifacts,
  author = {Nicole Meister and Dora Zhao and Angelina Wang and Vikram V. Ramaswamy and Ruth Fong and Olga Russakovsky},
  title = {Gender Artifacts in Visual Datasetsi},
  journal = {CoRR},
  volume = {abs/2206.09191},
  year={2022}
  }
```

## Setup

### Setup computing environment
```
conda create -n genderartifacts python=3.X.X. 
conda activate genderartifacts 
conda install --file requirements.txt
```

### Download data annotations
Download the annotations from the following sources and place them in ```data/{dataset_name}```. 
#### COCO
```wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip```

#### OpenImages
Follow the instructions from the [OpenImage website](https://storage.googleapis.com/openimages/web/extended.html) (copied below):

1. Download the downloader (open and press Ctrl + S), or directly run:

```wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py```

2. Run the following script, where $IMAGE_LIST_FILE is one of the files with image key lists above:

```python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER --num_processes=5```



## Experiments
### Resolution and Color
TBD 

### Person and Background
To generate the image manipulations in the paper, use the following scripts:

(* denotes available only for COCO)

| Name      | Script         | 
| ------------- |:-------------:| 
| Full NoBg     | ```python image_manipulations.py --type full``` | 
| MaskSegm*     | ```python image_manipulations.py --type segm --background```    |   
| MaskRect      | ```python image_manipulations.py --type rect --background```     |   
| MaskSegm NoBg*| ```python image_manipulations.py --type segm```             |
| MaskRect NoBg | ```python image_manipulations.py --type rect```              |

Note: make sure to specify the arguments --dataset $DATA --filepath $PATH --annotations $ANN --split $SPLIT as well. 

To train and evaluate the gender cue model, run the following scripts 

Train: ```bash train.sh X X ```

Evaluate: ``` bash eval.sh X X ```

### Contextual Objects
