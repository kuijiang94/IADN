# Decomposition Makes Better Rain Removal: An Improved Attention-guided Deraining Network (IADN)

This is an implementation of the IADN model proposed in the paper
([Decomposition Makes Better Rain Removal: An Improved Attention-guided Deraining Network])
with TensorFlow.

# Requirements

- Python 3
- TensorFlow 1.12.0
- OpenCV
- tqdm
- glob
- sys

# Usage

## I. Train the IADN model

### Dataset Organization Form

If you prepare your own dataset, please follow the following form:
|--train_data  

    |--rainysamples  
        |--file1
                ：  
        |--file2
            :
        |--filen
        
    |--clean samples
        |--file1
                ：  
        |--file2
            :
        |--filen
Then you can produce the corresponding '.npy' in the '/dataset/train_data/npy' file.
```
$ python crop.py
$ python rescale.py
$ python preprocessing.py
```

### Training
Download training dataset ((raw images)[Baidu Cloud](https://pan.baidu.com/s/1usedYAf3gYOgAJJUDlrwWg), (**Password:4qnh**) (.npy)[Baidu Cloud](https://pan.baidu.com/s/1hOmO-xrZ2I6sI4lXiqhStA), (**Password:gd2s**)), or prepare your own dataset like above form.

Run the following commands:
```
cd ./model
python train_IADN.py 
```

## II. Test the IADN model 

####  Test the Retraining Model With Your Own Dataset (TEST_IADN.PY)
Download the commonly used testing rain dataset (R100H, R100L, TEST100, TEST1200, TEST2800) ([Google Drive](https://drive.google.com/file/d/1H6kigSTD0mucIoXOhpXZn3UqYytpS4TX/view?usp=sharing)), and the test samples and the labels of joint tasks form (BDD350, COCO350, BDD150) ([Baidu Cloud](https://pan.baidu.com/s/1xA4kgSyi9vZxVAeGRvc1tw), (**Password:0e7o**)). 
In addition, the test results of other competing models can be downloaded from here ([TEST1200, TEST100](https://drive.google.com/file/d/11nKUDRWJuapT8rogr6FARCMJF3rJoJtE/view?usp=sharing), [R100H, R100L](https://drive.google.com/file/d/1An5OChbJZnkhlbwGIDQ7wDh-xpkbELp9/view?usp=sharing)).

Put your dataset in './test/test_data/*' (* denotes one of the tasks in [deraing, dehazing, and low-light enhancement]).

Select the special task, and then change the data path. Run the following commands:
```
cd ./model/test
python test_IADN.py
```
The deraining results will be in './test/test_data/*/IADN'.



# Citation
```
@InProceedings{Kui_2020_IADN,
	author = {Jiang, Kui and Wang, Zhongyuan and Yi, Peng and Han, Zhen and Lu, Tao and Huang, Baojin and Jiang, Junjun},
	title = {Decomposition Makes Better Rain Removal: An Improved Attention-guided Deraining Network},
	year = {2020}
}
```