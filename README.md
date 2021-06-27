# Ripple Video Prediction

## Dataset

Put all the base images under the folder `./images`, then safely execute the script `ripple.py` to do the dataset preparation, with default settings. Functions in this script are also implemented to finish the prediction job when the continous height field information is obtained. Please read it for more details. 

## Model Architecture 

This net takes one frame from ripple video as input, then predict the height field mask of it. 

    python main.py

Try to run the above command, the default trainning process is triggered.

## Results

Here's the architecture and training curve of image `0.png`, as well as some intermediate figures. (All can be found in folder `./output`)