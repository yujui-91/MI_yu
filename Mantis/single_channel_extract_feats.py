import numpy as np
import torch
import torch.nn.functional as F
from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

def resize(X):
    """
    將輸入訊號調整為 MantisV2 要求的 (n_samples, 1, 512)
    """
    # 如果輸入是 2D (n_samples, seq_len)，擴增維度變為 3D (n_samples, 1, seq_len)
    if len(X.shape) == 2:
        X = X[:, np.newaxis, :]
    
    # 轉換為 Tensor 並進行線性插值調整長度至 512
    X_tensor = torch.tensor(X, dtype=torch.float)
    X_scaled = F.interpolate(X_tensor, size=512, mode='linear', align_corners=False)
    
    return X_scaled.numpy()

def get_extract_model(layer_idx=0, device='cuda'):
    """
    初始化 MantisV2 模型並讀取權重
    """
    # 初始化架構
    network = MantisV2(device=device, return_transf_layer=layer_idx, output_token='combined')
    # 載入預訓練權重
    network = network.from_pretrained("paris-noah/MantisV2")
    
    # 初始化 Trainer
    model = MantisTrainer(device=device, network=network)
    
    return model

# 這裡不放任何 np.load 或 model.transform 的執行程式碼
# 這樣 import 時才不會報錯