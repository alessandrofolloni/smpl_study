import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from multiview_smplx_model import MultiviewSMPLXModel
from PIL import Image
import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
else:
    device = torch.device('cpu')
    print("MPS device not found. Using CPU")


class MultiviewDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): List of tuples. Each tuple contains:
                              - List of image paths for the four views
                              - Path to the SMPL-X parameters file (JSON)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_paths, params_path = self.data_list[idx]
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        # Load SMPL-X parameters
        with open(params_path, 'r') as f:
            params_dict = json.load(f)
        # Convert parameters to a flat tensor
        params = smplx_params_to_tensor(params_dict)
        return images, params


def smplx_params_to_tensor(params_dict):
    """
    Converts SMPL-X parameters dictionary to a flat tensor.

    Args:
        params_dict (dict): Dictionary containing SMPL-X parameters.

    Returns:
        torch.Tensor: Flattened tensor of SMPL-X parameters.
    """
    params = []
    # Translation
    params.extend(params_dict['transl'])
    # Global Orientation
    params.extend(rotation_matrix_to_rotvec(params_dict['global_orient'][0]))
    # Body Pose
    for rot_mat in params_dict['body_pose']:
        params.extend(rotation_matrix_to_rotvec(rot_mat))
    # Betas
    params.extend(params_dict['betas'])
    # Left Hand Pose
    for rot_mat in params_dict['left_hand_pose']:
        params.extend(rotation_matrix_to_rotvec(rot_mat))
    # Right Hand Pose
    for rot_mat in params_dict['right_hand_pose']:
        params.extend(rotation_matrix_to_rotvec(rot_mat))
    # Jaw Pose
    params.extend(rotation_matrix_to_rotvec(params_dict['jaw_pose'][0]))
    # Left Eye Pose
    params.extend(rotation_matrix_to_rotvec(params_dict['leye_pose'][0]))
    # Right Eye Pose
    params.extend(rotation_matrix_to_rotvec(params_dict['reye_pose'][0]))
    # Expression
    params.extend(params_dict['expression'])
    return torch.tensor(params, dtype=torch.float32)


def rotation_matrix_to_rotvec(rot_mat_list):
    """
    Converts a rotation matrix to a rotation vector (axis-angle).

    Args:
        rot_mat_list (list): Rotation matrix as a nested list.

    Returns:
        list: Rotation vector as a list of 3 floats.
    """
    rot_mat_np = np.array(rot_mat_list, dtype=np.float32)
    rotvec, _ = cv2.Rodrigues(rot_mat_np)
    return rotvec.flatten().tolist()


def tensor_to_smplx_params(params):
    """
    Converts a flat parameter tensor back into a structured SMPL-X parameters dictionary.

    Args:
        params (np.ndarray): Flattened parameters array.

    Returns:
        dict: SMPL-X parameters dictionary.
    """
    idx = 0
    params_dict = {}

    # Translation
    params_dict['transl'] = params[idx:idx+3].tolist()
    idx += 3

    # Global Orientation
    params_dict['global_orient'] = [rotvec_to_rotation_matrix(params[idx:idx+3])]
    idx += 3

    # Body Pose
    body_pose = []
    for _ in range(21):  # 21 body joints
        rot_mat = rotvec_to_rotation_matrix(params[idx:idx+3])
        body_pose.append(rot_mat)
        idx += 3
    params_dict['body_pose'] = body_pose

    # Betas
    params_dict['betas'] = params[idx:idx+10].tolist()
    idx += 10

    # Left Hand Pose
    left_hand_pose = []
    for _ in range(15):  # 15 hand joints
        rot_mat = rotvec_to_rotation_matrix(params[idx:idx+3])
        left_hand_pose.append(rot_mat)
        idx += 3
    params_dict['left_hand_pose'] = left_hand_pose

    # Right Hand Pose
    right_hand_pose = []
    for _ in range(15):  # 15 hand joints
        rot_mat = rotvec_to_rotation_matrix(params[idx:idx+3])
        right_hand_pose.append(rot_mat)
        idx += 3
    params_dict['right_hand_pose'] = right_hand_pose

    # Jaw Pose
    params_dict['jaw_pose'] = [rotvec_to_rotation_matrix(params[idx:idx+3])]
    idx += 3

    # Left Eye Pose
    params_dict['leye_pose'] = [rotvec_to_rotation_matrix(params[idx:idx+3])]
    idx += 3

    # Right Eye Pose
    params_dict['reye_pose'] = [rotvec_to_rotation_matrix(params[idx:idx+3])]
    idx += 3

    # Expression
    params_dict['expression'] = params[idx:idx+10].tolist()
    idx += 10

    return params_dict


def rotvec_to_rotation_matrix(rotvec):
    """
    Converts a rotation vector (axis-angle) to a rotation matrix.

    Args:
        rotvec (np.ndarray): Rotation vector of shape (3,).

    Returns:
        list: Rotation matrix as a nested list.
    """
    rotvec = rotvec.reshape((3, 1))
    rot_mat, _ = cv2.Rodrigues(rotvec)
    return rot_mat.tolist()


def save_smplx_parameters(params_tensor, filename):
    """
    Saves the SMPL-X parameters to a JSON file.

    Args:
        params_tensor (torch.Tensor): Tensor of SMPL-X parameters.
        filename (str): Path to save the JSON file.
    """
    params = params_tensor.detach().cpu().numpy()
    params_dict = tensor_to_smplx_params(params)
    with open(filename, 'w') as f:
        json.dump(params_dict, f, indent=4)


def evaluate_and_save_predictions(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            # Forward pass
            outputs = model(images)
            # Save the predicted parameters
            for i in range(outputs.size(0)):
                params_tensor = outputs[i]
                filename = f'predicted_params_batch{batch_idx}_sample{i}.json'
                save_smplx_parameters(params_tensor, filename)


def main():
    # Hyperparameters
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    num_views = 4
    smplx_param_size = 188

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize with ImageNet mean and std
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Prepare dataset
    # Example data_list: [([img1_view1, img1_view2, img1_view3, img1_view4], params1.json), ...]
    data_list = load_data_list()

    # Split dataset
    train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)

    train_dataset = MultiviewDataset(train_list, transform=transform)
    val_dataset = MultiviewDataset(val_list, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = MultiviewSMPLXModel(num_views=num_views, smplx_param_size=smplx_param_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, params) in enumerate(train_loader):
            # Move data to device
            images = [img.to(device) for img in images]
            params = params.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, params)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, params in val_loader:
                images = [img.to(device) for img in images]
                params = params.to(device)
                outputs = model(images)
                loss = criterion(outputs, params)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'multiview_smplx_model.pth')

    evaluate_and_save_predictions(model, val_loader)


def load_data_list():
    """
    Load your data list from your dataset.
    Returns:
        data_list (list): List of tuples. Each tuple contains:
                          - List of image paths for the four views
                          - Path to the SMPL-X parameters file (JSON)
    """
    data_root = '/path/to/dataset'  # Replace with the actual path to your dataset
    data_list = []

    # Get a list of all sample directories
    sample_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    sample_dirs.sort()  # Optional: sort the directories for consistency

    for sample_dir in sample_dirs:
        # Paths to the four images
        image_paths = [
            os.path.join(sample_dir, 'view_1.jpg'),
            os.path.join(sample_dir, 'view_2.jpg'),
            os.path.join(sample_dir, 'view_3.jpg'),
            os.path.join(sample_dir, 'view_4.jpg')
        ]

        # Check if all images exist
        if not all(os.path.exists(p) for p in image_paths):
            print(f"Missing images in {sample_dir}")
            continue  # Skip this sample if images are missing

        # Path to the SMPL-X parameters file
        params_path = os.path.join(sample_dir, 'smplx_params.json')
        if not os.path.exists(params_path):
            print(f"Missing SMPL-X parameters file in {sample_dir}")
            continue  # Skip this sample if parameters file is missing

        # Add to the data list
        data_list.append((image_paths, params_path))

    print(f"Total samples loaded: {len(data_list)}")
    return data_list


if __name__ == '__main__':
    main()