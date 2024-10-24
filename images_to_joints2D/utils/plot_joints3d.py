import os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


def plot_3d_joints(joints, exercise_key, frame_idx, output_dir='3d_plots', save=True):
    """
    Plot 3D joints and save the plot to the specified directory with exercise key and frame number.

    Parameters:
        joints (np.ndarray): A 25x3 array containing the 3D coordinates of joints.
        exercise_key (str): A string identifier for the exercise (used in filename).
        frame_idx (int): The frame index to use in the output filename.
        output_dir (str): The directory to save the output plot (default is '3d_plots').

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if joints.shape != (25, 3):
        raise ValueError(f"Expected shape (25, 3) for joints, but got {joints.shape}")

    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the joints
    ax.scatter(x, y, z, c='r', marker='o')

    # Define the connections between joints based on the updated joint mapping
    connections = [
        # Head and Face
        (10, 9),  # Head to Nose
        (9, 8),  # Nose to Neck

        # Torso
        (8, 7),  # Neck to Stomach
        (7, 0),  # Stomach to Central Hip
        (0, 1),  # Central Hip to Left Hip
        (0, 4),  # Central Hip to Right Hip

        # Right Arm
        (8, 14), (14, 15), (15, 16),  # Neck to Right Shoulder to Right Elbow to Right Wrist
        (16, 23), (16, 24),  # Right Wrist to Right Palm and Fingers

        # Left Arm
        (8, 11), (11, 12), (12, 13),  # Neck to Left Shoulder to Left Elbow to Left Wrist
        (13, 21), (13, 22),  # Left Wrist to Left Palm and Fingers

        # Right Leg
        (4, 5), (5, 6), (6, 19), (6, 20),  # Right Hip to Right Knee to Right Ankle to Right Toe

        # Left Leg
        (1, 2), (2, 3), (3, 17), (3, 18)  # Left Hip to Left Knee to Left Ankle to Left Toe
    ]

    # Plot the lines connecting the joints
    for idx1, idx2 in connections:
        ax.plot(
            [x[idx1], x[idx2]],
            [y[idx1], y[idx2]],
            [z[idx1], z[idx2]],
            c='b'
        )

    # Annotate each joint with its index number
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{i}', size=10, zorder=1, color='k')

    # Set equal axis scaling for better visualization
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set labels and adjust the view angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)  # Adjust viewing angle for better understanding

    # Optionally, add a legend
    ax.legend()

    # Improve layout
    plt.tight_layout()
    if save:
        # Save the plot to a JPEG file with the exercise key and frame index in the filename
        output_path = os.path.join(output_dir, f'{exercise_key}_frame_{frame_idx}.jpg')
        plt.savefig(output_path, format='jpg')

    plt.close(fig)
    print(f"Saved {exercise_key} frame {frame_idx}")