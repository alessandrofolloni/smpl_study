# SMPL_study

This repository contains the code for a comprehensive study on SMPL (and SMPLX) models, their parametrization and visualization.

## Table of Contents

- [Files](#files)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Files

Step-by-step instructions about the utility of every file.

- ### yolo_keypoints_demo.py 
  From Fit3D dataset, I selected an exercise of a given subject as an example, in particular band_pull_apart.
  
  The camera parameters are read and divided between extrisics and intrisics.
  As a model, I use YOLO to keep it simple and have positive results immediately.
  
  A while loop reads frames from the video until the end. For each frame:
  - It checks if a frame is successfully read.
  - It undistorts the frame using the undistort_image() function. The undistorted frame is added to the frames list. The index of the frame is added to the frame_indices list.
  - When batch_size frames are collected, they are processed using process_batch(), and the lists are reset for the next batch.
  
  In the end, it saves the keypoints in a JSON file.
  
  
