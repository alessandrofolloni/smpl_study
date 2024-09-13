import torch

# Supponiamo che il tensor di partenza sia di dimensioni [21, 3, 3]
body_pose = torch.rand(21, 3, 3)

# Appiattisci il tensor da [21, 3, 3] a [21, 9]
body_pose_flattened = body_pose.view(21, -1)

# Se hai bisogno di [1, 63], concatenando tutti i valori in una sola riga
body_pose_final = body_pose_flattened.view(1, -1)

print(f"Dimensioni iniziali: {body_pose.shape}")
print(f"Dimensioni dopo flattening: {body_pose_flattened.shape}")
print(f"Dimensioni finali: {body_pose_final.shape}")