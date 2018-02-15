Readme for building the app in AndroidSignClassifier/android-OpenCV

# Running prebuilt deep neural network model
## Preliminary 
1. Install python3

2. Install pip3

3. Install OpenCV for python
```
pip3 install opencv-python
```
4. Install tensorflow for python 
```
pip3 install tensorflow

```
## Extract candidates from image
1. Run opencvimpro.py
```
python3 opencvimpro.py
```
optional args:<br />
--image <br />
Path to desired image (default is test.jpg)


## Running a frozen model on an image
1. run the inference

```
python3 runinference.py
```

optional args: 
<br />
--frozen_model_filename<br />
specifies the path to a model, leave to run the newest graph(modelsnewIII)<br />
--image<br />
specifies a path to an image leave to use default images/validation/80/802(0).jpeg as input (only 20x20 images will actually run)
