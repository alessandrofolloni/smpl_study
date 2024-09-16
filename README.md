# SMPL_study

This repository contains the code for a comprehensive study on SMPL (and SMPLX) models, their parametrization and visualization.

## Table of Contents

- [Files](#files)


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

- ### trimesh_prova.py
  The script loads the neutral SMPLX body model and the SMPLX parametrization of band_pull_apart exercise as a JSON.

  ```
  smplx_model = smplx.create(model_path, model_type='smplx',
                           gender='neutral',
                           use_face_contour=False,
                           ext='npz')
  ```
  The given function creates the smplx_model object in python using the provided SMPLX body model.

  The comment section describes the parameters available in the JSON file and used by the SMPL-X model:
  
  - Main Parameters:
    - transl: Translation of the model in 3D space.
    - global_orient: Global orientation of the model.
    - body_pose: Pose of the body (excluding hands, face).
    - betas: Shape parameters affecting body morphology.
    
  - Additional Parameters:
    - left_hand_pose and right_hand_pose: Poses of the hands.
    - jaw_pose: Pose of the jaw.
    - leye_pose and reye_pose: Poses of the left and right eyes.
    - expression: Facial expression parameters.

  I load for effective visualization standard parameters into the smplx model then they are iteratively modified to understand the
  contribution of each one. For each parameter, it creates a modified 3D mesh and visualizes it using pyrender while also saving the mesh to
  an .obj file.
  
- ### visualize.ipynb
  This Python script is designed to read and process data from the FIT3D dataset.
  - Details:
    - dataset_name: Specifies the name of the dataset, in this case, “FIT3D”.
	- data_root: Root directory where the dataset is located.
	-	subset: Specifies the subset of the dataset, such as ‘train’ or ‘test’.
	-	subj_name: Subject identifier in the dataset (s03 in this case).
	-	action_name: Name of the action to be processed (band_pull_apart in this case).
	-	camera_name: Identifier of the camera view to be processed; different cameras may capture different perspectives of the same action.
	-	subject: Specifies the type of data; here, it indicates that the data includes markers (w_markers).
  
  - read_video(vid_path):
  This function reads a video file frame by frame and converts each frame from BGR (default in OpenCV) to RGB format (common in image processing libraries). The function:
    - Opens the video file specified by vid_path.
    - Reads frames in a loop until the end of the video.
    - Converts frames to RGB format and stores them in a list.
    - Closes the video file and returns the list of frames as a NumPy array.

