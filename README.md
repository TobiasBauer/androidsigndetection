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
## Running a frozen model
1. run the inference
optional args: --frozen_model_filename,  specifies the path to a model, leave to run the newest graph(modelsnewIII)
               --image, specifies a path to an image leave to use default images/cropped/10/10.jpg as input (only 20x20 images will actually run)
```
python3 runinference.py
```
