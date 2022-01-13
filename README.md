# Structure from Motion: Classical Approach
Python solution for [project 3: CMSC733 Computer vision course](https://cmsc733.github.io/2019/proj/p3/).
## Introduction
In this project, we reconstructed a 3D scene and simultaneously obtained the camera poses with respect to the scene,
with a given set of 6 images from a monocular camera and
their feature point correspondences. Following are the steps involved:
* Feature detection and finding correspondences
* Estimating Fundamental Matrix
* Essential Matrix and solving for camera poses
* Linear Triangulation and recovering correct pose
* Non Linear Triangulation
* Linear PnP, RANSAC and Non linear optimization
* Bundle Adjustment

Please refer the [report](https://github.com/sakshikakde/SFM/blob/master/Report-compressed.pdf) for detailed steps.

## Pipeline

![Pipeline](https://github.com/sakshikakde/SFM/blob/master/images/pipeline.png)


## How to run the code
- Change the directory to the folder where Wrapper.py is located. Eg.     
```
cd ./Code 
```

- Run the .py file using the following command:    
```
python3 Wrapper.py --DataPath ./Data/
```

## Parameters
- `--DataPath`: the path where the data is stored
- `--savepath`: the path to folder where the output will be saved
- `--BA`: True-If we want to use bundle adjustment

## Result

![Result image](https://github.com/sakshikakde/SFM/blob/master/images/result.png)



