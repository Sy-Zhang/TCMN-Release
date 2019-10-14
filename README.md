# Exploiting Temporal Relationships in Video Moment Localization with Natural Language


Songyang Zhang, Jinsong Su, Jiebo Luo, Exploiting Temporal Relationships in Video Moment Localization with Natural Language, In ACM Multimedia 2019

[arxiv preprint](https://arxiv.org/abs/1908.03846)

## Introduction

<img align="right" width="400" height="400" src="figures/illustration.png">

Moment localization with temporal language aims to locate a segment in a video referred to by temporal language, which describes relationships between multiple events in a video. It requires the model to be capable of localizing a single event and reasoning among multiple events. In the right figure, for example, the description *kitten paws at before the bottle is dropped* is composed of a main event, *kitten paws at the bottle*, a context event *the bottle is dropped*, and their temporal ordering *before*.

In this work, 
- We propose a novel model called Temporal Compositional Modular Network (TCMN) that first learns to softly decompose asentence into three descriptions with respect to the main event, context event and temporal signal, and then guides cross-modal feature matching by measuring the visual similarity and location similarity between each segment and the decomposed descriptions.
- we  further  form  an  ensemble  model  to  handle  multiple events that may reflect on different visual modalities.

## Main Results

##### Main result on TEMPO-HL
<table>
  <tr>
    <td colspan="2" align="center">DiDeMo</td>
    <td colspan="2" align="center">Before</td>
    <td colspan="2" align="center">After</td>
    <td colspan="2" align="center">Then</td>
    <td colspan="2" align="center">While</td>
    <td colspan="3" align="center">Average</td>
  </tr>
  <tr>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>mIoU</td>
  </tr>
  <tr>
    <td>28.77</td>
    <td>42.37</td>
    <td>35.47</td>
    <td>59.28</td>
    <td>17.91</td>
    <td>40.79</td>
    <td>20.47</td>
    <td>50.78</td>
    <td>18.81</td>
    <td>42.95</td>
    <td>24.29</td>
    <td>76.98</td>
    <td>47.24</td>
  </tr>
</table>

##### Main result on TEMPO-TL

<table>
  <tr>
    <td colspan="2" align="center">DiDeMo</td>
    <td colspan="2" align="center">Before</td>
    <td colspan="2" align="center">After</td>
    <td colspan="2" align="center">Then</td>
    <td colspan="3" align="center">Average</td>
  </tr>
  <tr>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>mIoU</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>mIoU</td>
  </tr>
  <tr>
    <td>28.90</td>
    <td>41.03</td>
    <td>37.68</td>
    <td>44.78</td>
    <td>32.61</td>
    <td>42.77</td>
    <td>31.16</td>
    <td>55.46</td>
    <td>32.85</td>
    <td>78.73</td>
    <td>46.01</td>
  </tr>
</table>

## Quick Start

### Prerequisites

There are a few dependencies to run the code. The major libraries we use are

- [pytorch](https://pytorch.org)
- [nltk](https://www.nltk.org/)
- [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser)
- [torchtext](https://github.com/pytorch/text)

The codes are written in Python3.

### Data Preparation

All video features are provided by [DiDeMo](https://github.com/LisaAnne/LocalizingMoments) and [TEMPO](https://github.com/LisaAnne/TemporalLanguageRelease).
Please download the feature under their instructions.

You can also run setup.sh to have a quick setup.

### Training Single Temporal Compositional Modular Network

```
CUDA_VISIBLE_DEVICES=0 python train.py -feature_type_0 rgb -feature_type_1 rgb -dataset_name TEMPO_HL -gpu 0 -vis_hidden_size 500 -lang_hidden_size 600 -att_hidden_size 250 -hidden_size 250 -batch_size 16 -verbose
CUDA_VISIBLE_DEVICES=1 python train.py -feature_type_0 flow -feature_type_1 rgb -dataset_name TEMPO_HL -gpu 0 -vis_hidden_size 500 -lang_hidden_size 600 -att_hidden_size 250 -hidden_size 250 -batch_size 16 -verbose
CUDA_VISIBLE_DEVICES=2 python train.py -feature_type_0 rgb -feature_type_1 flow -dataset_name TEMPO_HL -gpu 0 -vis_hidden_size 500 -lang_hidden_size 600 -att_hidden_size 250 -hidden_size 250 -batch_size 16 -verbose
CUDA_VISIBLE_DEVICES=3 python train.py -feature_type_0 flow -feature_type_1 flow -dataset_name TEMPO_HL -gpu 0 -vis_hidden_size 500 -lang_hidden_size 600 -att_hidden_size 250 -hidden_size 250 -batch_size 16 -verbose
```

### Testing Single Temporal Compositional Modular Network
Our model is provided [here](https://uofr-my.sharepoint.com/:u:/g/personal/szhang83_ur_rochester_edu/EXASlpqtG5FCm0UFIAEbVJMBSMjxDnU82Mrjp5PGVW1DIg?e=0NEHvd).

Please download them first, unzip to the ``checkpoints`` folder, and then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python test.py -feature_type_0 rgb -feature_type_1 rgb -batch_size 16 -hidden_size 250 -att_hidden_size 250 -vis_hidden_size 500 -lang_hidden_size 500 -dataset_name TEMPO_HL -split test -verbose
CUDA_VISIBLE_DEVICES=1 python test.py -feature_type_0 flow -feature_type_1 rgb -batch_size 16 -hidden_size 250 -att_hidden_size 250 -vis_hidden_size 500 -lang_hidden_size 500 -dataset_name TEMPO_HL -split test -verbose
CUDA_VISIBLE_DEVICES=2 python test.py -feature_type_0 rgb -feature_type_1 flow -batch_size 16 -hidden_size 250 -att_hidden_size 250 -vis_hidden_size 500 -lang_hidden_size 500 -dataset_name TEMPO_HL -split test -verbose
CUDA_VISIBLE_DEVICES=3 python test.py -feature_type_0 flow -feature_type_1 flow -batch_size 16 -hidden_size 250 -att_hidden_size 250 -vis_hidden_size 500 -lang_hidden_size 500 -dataset_name TEMPO_HL -split test -verbose
```
The result of the testing set will be output to the ``results`` folder.

You can also modify the model path in ``test.py`` to your trained model.

### Model Ensemble
Run the following command to get the ensemble result:
```
python late_fusion.py
```
