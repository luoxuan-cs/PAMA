PAMA
================
​		This is the Pytorch implementation of Progressive Attentional Manifold Alignment.

## Requirements

* python 3.6
* pytorch 1.8.0
* PIL, numpy, matplotlib

## Checkpoints

Please download the pre-trained checkpoints at [google drive](https://drive.google.com/file/d/1rPB_qnelVVSad6CtadmhRFi0PMI_RKdy/view?usp=sharing) and put them in ./checkpoints. 

Here we also provide some other pre-trained results with different loss weights at [google drive]()

## Training

The training set consists of two parts. We use WikiArt as style set and COCO2014 as content set.

```python
python main.py train --lr 1e-4 --content_folder ./COCO2014 --style_folder ./WikiArt
```

## Testing

To test the code, you need to specify the path to the image style and content. 

```python
python main.py eval --content ./content/1.jpg --style ./style/1.jpg
```

If you want to do a batch operation for all pictures under the folder at one time, please execute the following code.

```python
python main.py eval -- run_folder True --content ./content --style ./style
```


## Results Presentation

​		The results prove the quality of PAMA from three dimensions: Regional Consistency, Content Proservation, Style Quality.  
