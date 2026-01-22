import argparse

def get_trainer_args():
    """
    获取训练器参数的函数
    """
    parser = argparse.ArgumentParser(description='Rectified Flow Trainer Arguments')
    
    # 基本训练参数
    parser.add_argument('--num_train_steps', type=int, default=70000, 
                        help='训练步数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, 
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批处理大小')
    parser.add_argument('--data_type', type=str, default='image_label', 
                        choices=['image_label', 'image2image'],
                        help='数据类型：image_label用于图像标签条件，image2image用于图像到图像转换')
    
    # 文件路径参数
    parser.add_argument('--checkpoints_folder', type=str, default='./checkpoints', 
                        help='检查点保存文件夹')
    parser.add_argument('--results_folder', type=str, default='./results', 
                        help='结果保存文件夹')
    
    # 保存频率参数
    parser.add_argument('--save_results_every', type=int, default=100, 
                        help='保存结果的频率')
    parser.add_argument('--checkpoint_every', type=int, default=1000, 
                        help='保存检查点的频率')
    
    # 采样参数
    parser.add_argument('--sample_temperature', type=float, default=1.0, 
                        help='采样温度')
    parser.add_argument('--num_samples', type=int, default=16, 
                        help='采样数量')
    
    # EMA参数
    parser.add_argument('--use_ema', type=bool, default=True, 
                        help='是否使用指数移动平均')
    
    # 梯度参数
    parser.add_argument('--max_grad_norm', type=float, default=0.5, 
                        help='最大梯度范数')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_trainer_args()
    print("Trainer Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")