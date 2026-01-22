from rectified_flow_pytorch.rectified_flow import *
import os

# 验证CUDA_HOME设置
print(f"CUDA_HOME环境变量: {os.environ.get('CUDA_HOME', '未设置')}")
if os.path.exists(os.environ.get('CUDA_HOME', '')):
    print("CUDA_HOME路径存在")
    os.system('ls $CUDA_HOME')
else:
    print("警告: CUDA_HOME路径不存在!")


# 验证CUDA是否可用
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 验证CUDA_VISIBLE_DEVICES设置
    print(f"CUDA_VISIBLE_DEVICES环境变量: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
# print cureent cuda device gpu memory : 
print(f"当前CUDA设备内存: {torch.cuda.memory_allocated()}")
