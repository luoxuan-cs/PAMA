PAMA
================
​		This is the Pytorch implementation of Progressive Attentional Manifold Alignment.

## Requirements

* python 3.6
* pytorch 1.8.0
* PIL, numpy, scipy
* tqdm

## Checkpoints

​		Please download the pre-trained checkpoints at [google drive](https://drive.google.com/file/d/1rPB_qnelVVSad6CtadmhRFi0PMI_RKdy/view?usp=sharing) and put them in ./checkpoints. For other pre-trained results with different loss weights...（放之前让你训练的，后面可能还要跑一点）

## Training

​		The training set consists of two parts. Style set is WikiArt collected from WIKIART and content set is COCO2014.

```python
python main.py train
```

## Testing

```python
python main.py eval
```

## Results Presentation

​		The results prove the quality of PAMA from three dimensions: Regional Consistency, Content Proservation, Style Quality.  
