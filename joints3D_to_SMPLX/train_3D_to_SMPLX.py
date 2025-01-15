import os
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

configs = [
    # ===========================
    # Transformer Configurations (1-40)
    # ===========================

    # Transformer Configuration 1
    {
        'batch_size': 16,
        'learning_rate': 1e-5,
        'epochs': 50,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.1,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 2,4,8
        'transformer_nhead': 4,       # 128 / 4 = 32
        'transformer_num_layers': 2,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 2
    {
        'batch_size': 24,
        'learning_rate': 1.1e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.15,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 2,4,8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 3,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 3
    {
        'batch_size': 32,
        'learning_rate': 2e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6,4,3,2
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 4,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 4
    {
        'batch_size': 48,
        'learning_rate': 3e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8,4,2
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 5,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 5
    {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 2,4,8
        'transformer_nhead': 2,       # 128 / 2 = 64
        'transformer_num_layers': 6,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 6
    {
        'batch_size': 16,
        'learning_rate': 5e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 4,       # 256 / 4 = 64
        'transformer_num_layers': 7,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 7
    {
        'batch_size': 24,
        'learning_rate': 6e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 4,       # Adjusted from 6 to 4
        'transformer_num_layers': 5,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 8
    {
        'batch_size': 32,
        'learning_rate': 7e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 8,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 9
    {
        'batch_size': 48,
        'learning_rate': 8e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 9,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 10
    {
        'batch_size': 64,
        'learning_rate': 9e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 2,4,8
        'transformer_nhead': 2,       # 128 / 2 = 64
        'transformer_num_layers': 10,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 11
    {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.1,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 3,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 12
    {
        'batch_size': 24,
        'learning_rate': 1.1e-4,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 4,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 13
    {
        'batch_size': 32,
        'learning_rate': 1.2e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 5,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 14
    {
        'batch_size': 48,
        'learning_rate': 1.3e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 4,       # 128 / 4 = 32
        'transformer_num_layers': 6,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 15
    {
        'batch_size': 64,
        'learning_rate': 1.4e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 7,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 16
    {
        'batch_size': 16,
        'learning_rate': 1.5e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 8,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 17
    {
        'batch_size': 24,
        'learning_rate': 1.6e-4,
        'epochs': 95,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 9,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 18
    {
        'batch_size': 32,
        'learning_rate': 1.7e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 8,       # 128 / 8 = 16
        'transformer_num_layers': 10,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 19
    {
        'batch_size': 48,
        'learning_rate': 1.8e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 11,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 20
    {
        'batch_size': 64,
        'learning_rate': 1.9e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 12,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 21
    {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 55,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 13,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 22
    {
        'batch_size': 24,
        'learning_rate': 2.1e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 4,       # 128 / 4 = 32
        'transformer_num_layers': 14,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 23
    {
        'batch_size': 32,
        'learning_rate': 2.2e-4,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 15,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 24
    {
        'batch_size': 48,
        'learning_rate': 2.3e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 16,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 25
    {
        'batch_size': 64,
        'learning_rate': 2.4e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 8,       # 128 / 8 = 16
        'transformer_num_layers': 17,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 26
    {
        'batch_size': 16,
        'learning_rate': 2.5e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 4,8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 18,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 27
    {
        'batch_size': 24,
        'learning_rate': 2.6e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 19,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 28
    {
        'batch_size': 32,
        'learning_rate': 2.7e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 20,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 29
    {
        'batch_size': 48,
        'learning_rate': 2.8e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 8,       # 128 / 8 = 16
        'transformer_num_layers': 21,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 30
    {
        'batch_size': 64,
        'learning_rate': 2.9e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 22,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 31
    {
        'batch_size': 16,
        'learning_rate': 3e-4,
        'epochs': 55,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 23,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 32
    {
        'batch_size': 24,
        'learning_rate': 3.1e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 4,       # 128 / 4 = 32
        'transformer_num_layers': 24,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 33
    {
        'batch_size': 32,
        'learning_rate': 3.2e-4,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 256,   # Divisible by 8
        'transformer_nhead': 8,       # 256 / 8 = 32
        'transformer_num_layers': 25,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 34
    {
        'batch_size': 48,
        'learning_rate': 3.3e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 26,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 35
    {
        'batch_size': 64,
        'learning_rate': 3.4e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 27,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 36
    {
        'batch_size': 16,
        'learning_rate': 3.5e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 8,       # 128 / 8 = 16
        'transformer_num_layers': 28,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 37
    {
        'batch_size': 24,
        'learning_rate': 3.6e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 29,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 38
    {
        'batch_size': 32,
        'learning_rate': 3.7e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 512,   # Divisible by 8
        'transformer_nhead': 8,       # 512 / 8 = 64
        'transformer_num_layers': 30,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 39
    {
        'batch_size': 48,
        'learning_rate': 3.8e-4,
        'epochs': 95,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 128,   # Divisible by 4,8
        'transformer_nhead': 8,       # 128 / 8 = 16
        'transformer_num_layers': 31,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # Transformer Configuration 40
    {
        'batch_size': 64,
        'learning_rate': 4e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'Transformer',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 384,   # Divisible by 6
        'transformer_nhead': 6,       # 384 / 6 = 64
        'transformer_num_layers': 32,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # ===========================
    # RNN Configurations (41-60)
    # ===========================

    # RNN Configuration 41
    {
        'batch_size': 16,
        'learning_rate': 1e-5,
        'epochs': 50,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.1,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 128,
        'rnn_num_layers': 1,
    },

    # RNN Configuration 42
    {
        'batch_size': 24,
        'learning_rate': 1.1e-4,
        'epochs': 55,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.15,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 2,
    },

    # RNN Configuration 43
    {
        'batch_size': 32,
        'learning_rate': 1.2e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 3,
    },

    # RNN Configuration 44
    {
        'batch_size': 48,
        'learning_rate': 1.3e-4,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 1024,
        'rnn_num_layers': 4,
    },

    # RNN Configuration 45
    {
        'batch_size': 64,
        'learning_rate': 1.4e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 128,
        'rnn_num_layers': 1,
    },

    # RNN Configuration 46
    {
        'batch_size': 16,
        'learning_rate': 1.5e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 2,
    },

    # RNN Configuration 47
    {
        'batch_size': 24,
        'learning_rate': 1.6e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 3,
    },

    # RNN Configuration 48
    {
        'batch_size': 32,
        'learning_rate': 1.7e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 1024,
        'rnn_num_layers': 4,
    },

    # RNN Configuration 49
    {
        'batch_size': 48,
        'learning_rate': 1.8e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 128,
        'rnn_num_layers': 3,
    },

    # RNN Configuration 50
    {
        'batch_size': 64,
        'learning_rate': 1.9e-4,
        'epochs': 95,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 4,
    },

    # RNN Configuration 51
    {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 5,
    },

    # RNN Configuration 52
    {
        'batch_size': 24,
        'learning_rate': 2.1e-4,
        'epochs': 105,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 128,
        'rnn_num_layers': 4,
    },

    # RNN Configuration 53
    {
        'batch_size': 32,
        'learning_rate': 2.2e-4,
        'epochs': 110,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 6,
    },

    # RNN Configuration 54
    {
        'batch_size': 48,
        'learning_rate': 2.3e-4,
        'epochs': 115,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 5,
    },

    # RNN Configuration 55
    {
        'batch_size': 64,
        'learning_rate': 2.4e-4,
        'epochs': 120,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 128,
        'rnn_num_layers': 5,
    },

    # RNN Configuration 56
    {
        'batch_size': 16,
        'learning_rate': 2.5e-4,
        'epochs': 125,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 7,
    },

    # RNN Configuration 57
    {
        'batch_size': 24,
        'learning_rate': 2.6e-4,
        'epochs': 130,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 7,
    },

    # RNN Configuration 58
    {
        'batch_size': 32,
        'learning_rate': 2.7e-4,
        'epochs': 135,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 1024,
        'rnn_num_layers': 7,
    },

    # RNN Configuration 59
    {
        'batch_size': 48,
        'learning_rate': 2.8e-4,
        'epochs': 140,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 512,
        'rnn_num_layers': 8,
    },

    # RNN Configuration 60
    {
        'batch_size': 64,
        'learning_rate': 2.9e-4,
        'epochs': 145,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'RNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 256,
        'rnn_num_layers': 8,
    },

    # ===========================
    # CNN Configurations (61-80)
    # ===========================

    # CNN Configuration 61
    {
        'batch_size': 16,
        'learning_rate': 1e-5,
        'epochs': 50,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.1,
        'cnn_extra_layers': [32, 64],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 62
    {
        'batch_size': 24,
        'learning_rate': 1.1e-4,
        'epochs': 55,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.15,
        'cnn_extra_layers': [64, 128],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 63
    {
        'batch_size': 32,
        'learning_rate': 1.2e-4,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [128, 256],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 64
    {
        'batch_size': 48,
        'learning_rate': 1.3e-4,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [256, 512],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 65
    {
        'batch_size': 64,
        'learning_rate': 1.4e-4,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [512, 1024],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 66
    {
        'batch_size': 16,
        'learning_rate': 1.5e-4,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [64, 128, 256],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 67
    {
        'batch_size': 24,
        'learning_rate': 1.6e-4,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [128, 256, 512],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 68
    {
        'batch_size': 32,
        'learning_rate': 1.7e-4,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [256, 512, 1024],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 69
    {
        'batch_size': 48,
        'learning_rate': 1.8e-4,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [512, 1024, 2048],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 70
    {
        'batch_size': 64,
        'learning_rate': 1.9e-4,
        'epochs': 95,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [1024, 2048, 4096],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 71
    {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.15,
        'cnn_extra_layers': [32, 64, 128, 256],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 72
    {
        'batch_size': 24,
        'learning_rate': 2.1e-4,
        'epochs': 105,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.2,
        'cnn_extra_layers': [64, 128, 256, 512],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 73
    {
        'batch_size': 32,
        'learning_rate': 2.2e-4,
        'epochs': 110,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.25,
        'cnn_extra_layers': [128, 256, 512, 1024],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 74
    {
        'batch_size': 48,
        'learning_rate': 2.3e-4,
        'epochs': 115,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.3,
        'cnn_extra_layers': [256, 512, 1024, 2048],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 75
    {
        'batch_size': 64,
        'learning_rate': 2.4e-4,
        'epochs': 120,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.35,
        'cnn_extra_layers': [512, 1024, 2048, 4096],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 76
    {
        'batch_size': 16,
        'learning_rate': 2.5e-4,
        'epochs': 125,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.4,
        'cnn_extra_layers': [64, 128, 256, 512, 1024],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 77
    {
        'batch_size': 24,
        'learning_rate': 2.6e-4,
        'epochs': 130,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.45,
        'cnn_extra_layers': [128, 256, 512, 1024, 2048],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 78
    {
        'batch_size': 32,
        'learning_rate': 2.7e-4,
        'epochs': 135,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [256, 512, 1024, 2048, 4096],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 79
    {
        'batch_size': 48,
        'learning_rate': 2.8e-4,
        'epochs': 140,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.5,
        'cnn_extra_layers': [512, 1024, 2048, 4096, 8192],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # CNN Configuration 80
    {
        'batch_size': 64,
        'learning_rate': 3e-4,
        'epochs': 145,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'CNN',
        'hidden_sizes': [],
        'dropout': 0.5,  # Adjusted from 1.05 to 0.5
        'cnn_extra_layers': [1024, 2048, 4096, 8192, 16384],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # ===========================
    # FCNN Configurations (81-100)
    # ===========================

    # FCNN Configuration 81
    {
        'batch_size': 16,
        'learning_rate': 1e-5,
        'epochs': 50,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [512, 256, 128],
        'dropout': 0.1,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 82
    {
        'batch_size': 24,
        'learning_rate': 1.1e-5,
        'epochs': 55,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.15,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 83
    {
        'batch_size': 32,
        'learning_rate': 1.2e-5,
        'epochs': 60,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [128, 64, 32],
        'dropout': 0.2,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 84
    {
        'batch_size': 48,
        'learning_rate': 1.3e-5,
        'epochs': 65,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [64, 32, 16],
        'dropout': 0.25,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 85
    {
        'batch_size': 64,
        'learning_rate': 1.4e-5,
        'epochs': 70,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [32, 16, 8],
        'dropout': 0.3,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 86
    {
        'batch_size': 16,
        'learning_rate': 1.5e-5,
        'epochs': 75,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [512, 256, 128, 64],
        'dropout': 0.35,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 87
    {
        'batch_size': 24,
        'learning_rate': 1.6e-5,
        'epochs': 80,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [256, 128, 64, 32],
        'dropout': 0.4,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 88
    {
        'batch_size': 32,
        'learning_rate': 1.7e-5,
        'epochs': 85,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [128, 64, 32, 16],
        'dropout': 0.45,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 89
    {
        'batch_size': 48,
        'learning_rate': 1.8e-5,
        'epochs': 90,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [64, 32, 16, 8],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 90
    {
        'batch_size': 64,
        'learning_rate': 1.9e-5,
        'epochs': 95,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [512, 256, 128, 64, 32],
        'dropout': 0.5,  # Adjusted from 1.0 to 0.5
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 91
    {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 100,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [256, 128, 64, 32, 16],
        'dropout': 0.5,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 92
    {
        'batch_size': 24,
        'learning_rate': 2.1e-5,
        'epochs': 105,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [128, 64, 32, 16, 8],
        'dropout': 0.55,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 93
    {
        'batch_size': 32,
        'learning_rate': 2.2e-5,
        'epochs': 110,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [64, 32, 16, 8, 4],
        'dropout': 0.6,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 94
    {
        'batch_size': 48,
        'learning_rate': 2.3e-5,
        'epochs': 115,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [512, 256, 128, 64, 32, 16],
        'dropout': 0.65,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 95
    {
        'batch_size': 64,
        'learning_rate': 2.4e-5,
        'epochs': 120,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [256, 128, 64, 32, 16, 8],
        'dropout': 0.7,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 96
    {
        'batch_size': 16,
        'learning_rate': 2.5e-5,
        'epochs': 125,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [128, 64, 32, 16, 8, 4],
        'dropout': 0.75,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 97
    {
        'batch_size': 24,
        'learning_rate': 2.6e-5,
        'epochs': 130,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [64, 32, 16, 8, 4, 2],
        'dropout': 0.8,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 98
    {
        'batch_size': 32,
        'learning_rate': 2.7e-5,
        'epochs': 135,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [512, 256, 128, 64, 32, 16, 8],
        'dropout': 0.85,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 99
    {
        'batch_size': 48,
        'learning_rate': 2.8e-5,
        'epochs': 140,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [256, 128, 64, 32, 16, 8, 4],
        'dropout': 0.9,
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },

    # FCNN Configuration 100
    {
        'batch_size': 64,
        'learning_rate': 3e-5,
        'epochs': 145,
        'camera_ids': ['58860488', '60457274', '65906101', '50591643'],
        'num_joints_3d': 25,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1,
        'model_name': 'FCNN',
        'hidden_sizes': [128, 64, 32, 16, 8, 4, 2],
        'dropout': 0.5,  # Adjusted from 1.05 to 0.5
        'cnn_extra_layers': [],
        'transformer_d_model': 0,
        'transformer_nhead': 0,
        'transformer_num_layers': 0,
        'rnn_hidden_size': 0,
        'rnn_num_layers': 0,
    },
]

# Function to generate a unique run name based on model parameters
def generate_run_name(config):
    model_type = config.get('model_name', 'UnknownModel')
    run_name = f"{model_type}_bs{config['batch_size']}_lr{config['learning_rate']}_epochs{config['epochs']}"
    if model_type == 'FCNN':
        hidden_sizes = config.get('hidden_sizes', [])
        hidden_sizes_str = '_'.join(map(str, hidden_sizes)) if hidden_sizes else 'None'
        run_name += f"_hidden_sizes_{hidden_sizes_str}_dropout_{config['dropout']}"
    elif model_type == 'CNN':
        cnn_layers = config.get('cnn_extra_layers', [])
        cnn_layers_str = '_'.join(map(str, cnn_layers)) if cnn_layers else 'None'
        run_name += f"_cnn_layers_{cnn_layers_str}_dropout_{config['dropout']}"
    elif model_type == 'Transformer':
        d_model = config.get('transformer_d_model', 'None')
        nhead = config.get('transformer_nhead', 'None')
        num_layers = config.get('transformer_num_layers', 'None')
        run_name += f"_dmodel_{d_model}_nhead_{nhead}_layers_{num_layers}_dropout_{config['dropout']}"
    elif model_type == 'RNN':
        hidden_size = config.get('rnn_hidden_size', 'None')
        num_layers = config.get('rnn_num_layers', 'None')
        run_name += f"_hidden_size_{hidden_size}_layers_{num_layers}_dropout_{config['dropout']}"
    else:
        run_name += "_unknown_params"
    return run_name

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define Dataset
class JointsToSMPLXDataset(Dataset):
    def __init__(self, mega_dict):
        self.samples = []
        required_keys = ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose',
                         'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
        for exercise_key, frames in mega_dict.items():
            for frame_key, frame_data in frames.items():
                gt_params = frame_data['gt']
                if all(key in gt_params for key in required_keys):
                    sample = {
                        'joints3d': frame_data['joints3d'],
                        'gt': gt_params,
                        'exercise_key': exercise_key,
                        'frame_idx': frame_key
                    }
                    self.samples.append(sample)
                else:
                    print(f"Skipping sample {exercise_key} frame {frame_key} due to missing SMPL-X parameters.")
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def rotation_matrices_to_axis_angles(self, rotation_matrices):
        rot_obj = R.from_matrix(rotation_matrices)
        axis_angles = rot_obj.as_rotvec()
        return axis_angles

    def __getitem__(self, idx):
        sample = self.samples[idx]
        joints3d = np.array(sample['joints3d'])  # Shape: (25, 3)
        gt_params = sample['gt']

        try:
            # Extract rotation matrices
            global_orient_mat = np.array(gt_params['global_orient'])  # (1, 3, 3)
            body_pose_mat = np.array(gt_params['body_pose'])  # (21, 3, 3)
            left_hand_pose_mat = np.array(gt_params['left_hand_pose'])  # (15, 3, 3)
            right_hand_pose_mat = np.array(gt_params['right_hand_pose']) # (15, 3, 3)
            jaw_pose_mat = np.array(gt_params['jaw_pose']) # (1, 3, 3)
            leye_pose_mat = np.array(gt_params['leye_pose']) # (1, 3, 3)
            reye_pose_mat = np.array(gt_params['reye_pose']) # (1, 3, 3)

            # Convert to axis-angle
            global_orient_axis_angle = self.rotation_matrices_to_axis_angles(global_orient_mat)
            body_pose_axis_angle = self.rotation_matrices_to_axis_angles(body_pose_mat)
            left_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(left_hand_pose_mat)
            right_hand_pose_axis_angle = self.rotation_matrices_to_axis_angles(right_hand_pose_mat)
            jaw_pose_axis_angle = self.rotation_matrices_to_axis_angles(jaw_pose_mat)
            leye_pose_axis_angle = self.rotation_matrices_to_axis_angles(leye_pose_mat)
            reye_pose_axis_angle = self.rotation_matrices_to_axis_angles(reye_pose_mat)

            transl = np.array(gt_params['transl'])  # (3,)
            betas = np.array(gt_params['betas'])  # (10,)
            expression = np.array(gt_params['expression'])  # (10,)

            # Flatten parameters
            gt_vector = np.concatenate([
                transl,
                global_orient_axis_angle.flatten(),
                body_pose_axis_angle.flatten(),
                betas,
                left_hand_pose_axis_angle.flatten(),
                right_hand_pose_axis_angle.flatten(),
                jaw_pose_axis_angle.flatten(),
                leye_pose_axis_angle.flatten(),
                reye_pose_axis_angle.flatten(),
                expression
            ])

            # Convert to tensors
            joints3d_tensor = torch.tensor(joints3d, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_vector, dtype=torch.float32)

            return joints3d_tensor, gt_tensor, sample['exercise_key'], sample['frame_idx']
        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")
            return self.__getitem__((idx + 1) % self.num_samples)

# Define Models
class FCNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout):
        super(FCNNModel, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size, cnn_channels, dropout):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels in cnn_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, features]
        x = self.cnn(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * 25, output_size)  # Assuming sequence_length = 25

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [sequence_length, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, sequence_length, d_model]
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Define Metrics
def compute_mae(predicted, target):
    mae = torch.mean(torch.abs(predicted - target))
    return mae.item()

def compute_mse(predicted, target):
    mse = torch.mean((predicted - target)**2)
    return mse.item()

def compute_rmse(predicted, target):
    rmse = torch.sqrt(torch.mean((predicted - target)**2))
    return rmse.item()

def compute_groupwise_mae(predicted, target):
    errors = {}
    # Define slices for each group
    transl_indices = slice(0, 3)
    global_orient_indices = slice(3, 6)
    body_pose_indices = slice(6, 69)      # 63 values -> 21 joints * 3
    betas_indices = slice(69, 79)          # 10 values
    left_hand_pose_indices = slice(79, 124)   # 45 values -> 15 * 3
    right_hand_pose_indices = slice(124, 169) # 45 values -> 15 * 3
    jaw_pose_indices = slice(169, 172)       # 3 values
    leye_pose_indices = slice(172, 175)      # 3 values
    reye_pose_indices = slice(175, 178)      # 3 values
    expression_indices = slice(178, 188)     # 10 values

    # Compute MAE for each group
    errors['transl'] = torch.mean(torch.abs(predicted[:, transl_indices] - target[:, transl_indices])).item()
    errors['global_orient'] = torch.mean(torch.abs(predicted[:, global_orient_indices] - target[:, global_orient_indices])).item()
    errors['body_pose'] = torch.mean(torch.abs(predicted[:, body_pose_indices] - target[:, body_pose_indices])).item()
    errors['betas'] = torch.mean(torch.abs(predicted[:, betas_indices] - target[:, betas_indices])).item()
    errors['left_hand_pose'] = torch.mean(torch.abs(predicted[:, left_hand_pose_indices] - target[:, left_hand_pose_indices])).item()
    errors['right_hand_pose'] = torch.mean(torch.abs(predicted[:, right_hand_pose_indices] - target[:, right_hand_pose_indices])).item()
    errors['jaw_pose'] = torch.mean(torch.abs(predicted[:, jaw_pose_indices] - target[:, jaw_pose_indices])).item()
    errors['leye_pose'] = torch.mean(torch.abs(predicted[:, leye_pose_indices] - target[:, leye_pose_indices])).item()
    errors['reye_pose'] = torch.mean(torch.abs(predicted[:, reye_pose_indices] - target[:, reye_pose_indices])).item()
    errors['expression'] = torch.mean(torch.abs(predicted[:, expression_indices] - target[:, expression_indices])).item()
    return errors

# Training, Validation, Testing Functions
def train_epoch(model, loader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    for batch in loader:
        inputs, targets, _, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Reshape inputs based on model type
        if config['model_name'] == 'RNN':
            # For RNN, inputs should be [batch, sequence_length, input_size]
            # Assuming joints3d is [batch, 25, 3]
            inputs = inputs.view(inputs.size(0), 25, 3)
        elif config['model_name'] == 'Transformer':
            # For Transformer, inputs should be [batch, sequence_length, input_size]
            inputs = inputs.view(inputs.size(0), 25, 3)
        elif config['model_name'] == 'FCNN':
            # For FCNN, flatten the inputs
            inputs = inputs.view(inputs.size(0), -1)
        elif config['model_name'] == 'CNN':
            # For CNN, treat input as [batch, channels, features]
            inputs = inputs.view(inputs.size(0), 3, 25)  # 3 channels (x,y,z), 25 joints
        else:
            raise ValueError(f"Unknown model name: {config['model_name']}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        mae = compute_mae(outputs, targets)
        mse = compute_mse(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_mae += mae * inputs.size(0)
        running_mse += mse * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = running_mae / len(loader.dataset)
    epoch_mse = running_mse / len(loader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)
    return epoch_loss, epoch_mae, epoch_mse, epoch_rmse

def validate_epoch(model, loader, criterion, config):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    groupwise_errors = {}
    with torch.no_grad():
        for batch in loader:
            inputs, targets, _, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reshape inputs based on model type
            if config['model_name'] == 'RNN':
                inputs = inputs.view(inputs.size(0), 25, 3)
            elif config['model_name'] == 'Transformer':
                inputs = inputs.view(inputs.size(0), 25, 3)
            elif config['model_name'] == 'FCNN':
                inputs = inputs.view(inputs.size(0), -1)
            elif config['model_name'] == 'CNN':
                inputs = inputs.view(inputs.size(0), 3, 25)
            else:
                raise ValueError(f"Unknown model name: {config['model_name']}")

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = compute_mae(outputs, targets)
            mse = compute_mse(outputs, targets)
            grp_errors = compute_groupwise_mae(outputs, targets)
            for key in grp_errors:
                if key not in groupwise_errors:
                    groupwise_errors[key] = 0.0
                groupwise_errors[key] += grp_errors[key] * inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_mae += mae * inputs.size(0)
            running_mse += mse * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = running_mae / len(loader.dataset)
    epoch_mse = running_mse / len(loader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)
    for key in groupwise_errors:
        groupwise_errors[key] /= len(loader.dataset)
    return epoch_loss, epoch_mae, epoch_mse, epoch_rmse, groupwise_errors

def test_model(model, loader, config, output_folder):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    groupwise_errors = {}
    predicted_params_list = []
    gt_params_list = []
    exercise_keys = []
    frame_idxs = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets, ex_keys, fr_idxs = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reshape inputs based on model type
            if config['model_name'] == 'RNN':
                inputs = inputs.view(inputs.size(0), 25, 3)
            elif config['model_name'] == 'Transformer':
                inputs = inputs.view(inputs.size(0), 25, 3)
            elif config['model_name'] == 'FCNN':
                inputs = inputs.view(inputs.size(0), -1)
            elif config['model_name'] == 'CNN':
                inputs = inputs.view(inputs.size(0), 3, 25)
            else:
                raise ValueError(f"Unknown model name: {config['model_name']}")

            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            mae = compute_mae(outputs, targets)
            mse = compute_mse(outputs, targets)
            grp_errors = compute_groupwise_mae(outputs, targets)
            for key in grp_errors:
                if key not in groupwise_errors:
                    groupwise_errors[key] = 0.0
                groupwise_errors[key] += grp_errors[key] * inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_mae += mae * inputs.size(0)
            running_mse += mse * inputs.size(0)
            # Collect predictions and ground truth for saving
            predicted_params_list.extend(outputs.cpu().numpy())
            gt_params_list.extend(targets.cpu().numpy())
            exercise_keys.extend(ex_keys)
            frame_idxs.extend(fr_idxs)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_mae = running_mae / len(loader.dataset)
    epoch_mse = running_mse / len(loader.dataset)
    epoch_rmse = np.sqrt(epoch_mse)
    for key in groupwise_errors:
        groupwise_errors[key] /= len(loader.dataset)

    # Save predictions
    predictions_path = os.path.join(output_folder, 'predictions.npz')
    np.savez(predictions_path,
             predicted_params=np.array(predicted_params_list),
             gt_params=np.array(gt_params_list),
             exercise_keys=np.array(exercise_keys),
             frame_idxs=np.array(frame_idxs))
    print(f"Predictions saved to {predictions_path}")

    return epoch_loss, epoch_mae, epoch_mse, epoch_rmse, groupwise_errors

# Rebuild SMPLX-like JSON from predicted parameters (if needed)
def rebuild_smplx_dict(predicted_params_list, exercise_keys, frame_idxs):
    # Define slices for parameters
    transl_indices = slice(0, 3)
    global_orient_indices = slice(3, 6)
    body_pose_indices = slice(6, 69)      # 63 values -> 21 joints * 3
    betas_indices = slice(69, 79)          # 10 values
    left_hand_pose_indices = slice(79, 124)   # 45 values -> 15 * 3
    right_hand_pose_indices = slice(124, 169) # 45 values -> 15 * 3
    jaw_pose_indices = slice(169, 172)       # 3 values
    leye_pose_indices = slice(172, 175)      # 3 values
    reye_pose_indices = slice(175, 178)      # 3 values
    expression_indices = slice(178, 188)     # 10 values

    predictions_by_exercise = {}
    for params, ex_key, fr_idx in zip(predicted_params_list, exercise_keys, frame_idxs):
        if ex_key not in predictions_by_exercise:
            predictions_by_exercise[ex_key] = {
                "transl": [],
                "global_orient": [],
                "body_pose": [],
                "betas": [],
                "left_hand_pose": [],
                "right_hand_pose": [],
                "jaw_pose": [],
                "leye_pose": [],
                "reye_pose": [],
                "expression": []
            }

        # Extract parameters
        transl = params[transl_indices]
        global_orient = params[global_orient_indices].reshape(1, 3)
        body_pose = params[body_pose_indices].reshape(21, 3)
        betas = params[betas_indices]
        left_hand_pose = params[left_hand_pose_indices].reshape(15, 3)
        right_hand_pose = params[right_hand_pose_indices].reshape(15, 3)
        jaw_pose = params[jaw_pose_indices].reshape(1, 3)
        leye_pose = params[leye_pose_indices].reshape(1, 3)
        reye_pose = params[reye_pose_indices].reshape(1, 3)
        expression = params[expression_indices]

        # Convert axis-angle back to rotation matrices
        global_orient_mat = R.from_rotvec(global_orient).as_matrix().tolist()
        body_pose_mat = R.from_rotvec(body_pose).as_matrix().tolist()
        left_hand_pose_mat = R.from_rotvec(left_hand_pose).as_matrix().tolist()
        right_hand_pose_mat = R.from_rotvec(right_hand_pose).as_matrix().tolist()
        jaw_pose_mat = R.from_rotvec(jaw_pose).as_matrix().tolist()
        leye_pose_mat = R.from_rotvec(leye_pose).as_matrix().tolist()
        reye_pose_mat = R.from_rotvec(reye_pose).as_matrix().tolist()

        # Append to exercise data
        predictions_by_exercise[ex_key]["transl"].append(transl.tolist())
        predictions_by_exercise[ex_key]["global_orient"].append(global_orient_mat)
        predictions_by_exercise[ex_key]["body_pose"].append(body_pose_mat)
        predictions_by_exercise[ex_key]["betas"].append(betas.tolist())
        predictions_by_exercise[ex_key]["left_hand_pose"].append(left_hand_pose_mat)
        predictions_by_exercise[ex_key]["right_hand_pose"].append(right_hand_pose_mat)
        predictions_by_exercise[ex_key]["jaw_pose"].append(jaw_pose_mat)
        predictions_by_exercise[ex_key]["leye_pose"].append(leye_pose_mat)
        predictions_by_exercise[ex_key]["reye_pose"].append(reye_pose_mat)
        predictions_by_exercise[ex_key]["expression"].append(expression.tolist())

    return predictions_by_exercise

# Main training loop
def main():
    # Load mega_dict
    mega_dict_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/train/mega_dict_3dSMPLX.json'
    with open(mega_dict_path, 'r') as f:
        mega_dict = json.load(f)

    for idx, config in enumerate(configs, 1):
        print(f"\n\n===== Starting Configuration {idx}/{len(configs)} =====")
        run_name = generate_run_name(config)
        print(f"Run Name: {run_name}")

        # Initialize wandb for this run
        wandb.init(
            project='3D_Joints_to_SMPLX_FINAL',
            entity='alessandrofolloni',
            config=config,
            name=run_name,
            reinit=True  # Important for multiple runs in a loop
        )

        # Create output directory
        output_folder = os.path.join('runs', run_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Outputs will be saved to: {output_folder}")

        # Prepare Dataset and DataLoaders
        dataset = JointsToSMPLXDataset(mega_dict)
        if len(dataset) == 0:
            print("Dataset is empty after filtering. Skipping this configuration.")
            wandb.finish()
            continue

        # Split dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(42)  # Ensure reproducibility
        np.random.shuffle(indices)

        train_split = int(np.floor(config['train_split'] * dataset_size))
        val_split = int(np.floor(config['val_split'] * dataset_size))
        test_split = dataset_size - train_split - val_split

        train_indices = indices[:train_split]
        val_indices = indices[train_split:train_split + val_split]
        test_indices = indices[train_split + val_split:]

        print(f"Dataset sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Initialize Model
        model_type = config.get('model_name', 'UnknownModel')
        output_size = 188  # As per your dataset structure

        if model_type == 'FCNN':
            input_size = config['num_joints_3d'] * 3  # Assuming joints3d has 3 coordinates
            hidden_sizes = config.get('hidden_sizes', [512, 256])
            model = FCNNModel(input_size, output_size, hidden_sizes, config.get('dropout', 0.0))
        elif model_type == 'CNN':
            input_channels = 3  # Assuming x, y, z channels
            cnn_channels = config.get('cnn_extra_layers', [64, 128, 256])
            model = CNNModel(input_channels, output_size, cnn_channels, config.get('dropout', 0.0))
        elif model_type == 'RNN':
            input_size = 3  # Each joint has 3 coordinates
            hidden_size = config.get('rnn_hidden_size', 512)
            num_layers = config.get('rnn_num_layers', 2)
            model = RNNModel(input_size, hidden_size, num_layers, output_size, config.get('dropout', 0.0))
        elif model_type == 'Transformer':
            input_size = 3  # Each joint has 3 coordinates
            d_model = config.get('transformer_d_model', 256)
            nhead = config.get('transformer_nhead', 8)
            num_layers = config.get('transformer_num_layers', 6)
            model = TransformerModel(input_size, output_size, d_model, nhead, num_layers, config.get('dropout', 0.0))
        else:
            print(f"Unknown model type: {model_type}. Skipping this configuration.")
            wandb.finish()
            continue

        model.to(device)
        print(f"Model '{model_type}' initialized and moved to device.")

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Training loop
        for epoch in range(config['epochs']):
            train_loss, train_mae, train_mse, train_rmse = train_epoch(model, train_loader, criterion, optimizer, config)
            val_loss, val_mae, val_mse, val_rmse, val_groupwise = validate_epoch(model, val_loader, criterion, config)

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                **{f'val_group_mae/{k}': v for k, v in val_groupwise.items()}
            })

            # Print epoch summary
            print(f"Epoch [{epoch + 1}/{config['epochs']}], "
                  f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

        # Testing
        test_loss, test_mae, test_mse, test_rmse, test_groupwise = test_model(model, test_loader, config, output_folder)

        # Log test metrics
        wandb.log({
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            **{f'test_group_mae/{k}': v for k, v in test_groupwise.items()}
        })

        print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
        print("Group-wise MAE on Test Set:")
        for key, value in test_groupwise.items():
            print(f"  {key}: {value:.4f}")

        # Save the model
        model_save_path = os.path.join(output_folder, f'model_{model_type}.pth')
        torch.save(model.state_dict(), model_save_path)
        wandb.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Rebuild SMPLX-like JSON structures from predicted params
        # (Optional: Implement if needed based on your application)
        # predictions_by_exercise = rebuild_smplx_dict(predicted_params_list, exercise_keys, frame_idxs)
        # for ex_key, ex_data in predictions_by_exercise.items():
        #     output_json_path = os.path.join(output_folder, f"{ex_key}_predicted_smplx.json")
        #     with open(output_json_path, 'w') as json_file:
        #         json.dump(ex_data, json_file, indent=4)
        #     print(f"Predicted SMPLX data saved to {output_json_path}")

        # Finish wandb run
        wandb.finish()

if __name__ == '__main__':
    main()