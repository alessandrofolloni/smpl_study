import torch
import torch.nn as nn
from images_to_smplx.image_encoder import ImageEncoder
from smplx_regressor import SMPLXRegressor

SMPLX_PARAM_SIZE = 188
'''Total Parameters: Translation: 3, Global Orientation: 3, Body Pose: 63, Shape (betas): 10
                    Left Hand Pose: 45, Right Hand Pose: 45, Jaw Pose: 3, Left Eye Pose: 3, Right Eye Pose: 3
                    Expression: 10

Sum: 3 + 3 + 63 + 10 + 45 + 45 + 3 + 3 + 3 + 10 = 188 parameters
'''


class MultiviewSMPLXModel(nn.Module):
    def __init__(self, num_views=4, smplx_param_size=SMPLX_PARAM_SIZE):
        super(MultiviewSMPLXModel, self).__init__()
        self.num_views = num_views
        self.encoders = nn.ModuleList([ImageEncoder() for _ in range(num_views)])
        self.regressor = SMPLXRegressor(input_size=2048 * num_views, output_size=smplx_param_size)

    def forward(self, images):
        features = []
        for i in range(self.num_views):
            feat = self.encoders[i](images[i])
            features.append(feat)
        # Concatenate features from all views
        fused_features = torch.cat(features, dim=1)
        # Predict SMPL-X parameters
        params = self.regressor(fused_features)
        return params