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
    
  - Additional Parameters:
    - left_hand_pose and right_hand_pose: Poses of the hands.
    - jaw_pose: Pose of the jaw.
    - leye_pose and reye_pose: Poses of the left and right eyes.
    - expression: Facial expression parameters.

  
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
  


