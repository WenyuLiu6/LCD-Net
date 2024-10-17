# LCD-Net: A Lightweight Remote Sensing Change Detection Network Combining Feature Fusion and Gating Mechanism
##
The proposed LCD-Net architecture.
![Image Description](Framework.png)
## Requirements

- Python 3.8
- Pytorch 1.8.0
- Torchvision 0.9.0
- OpenCV 4.5.3.56
- TensorboardX 2.4
- Cuda 11.3.1
- Cudnn 11.3

## Training and Test Process
```bash
python train_CGNet.py --epoch 50 --batchsize 8 --gpu_id '1' --data_name 'LEVIR' --model_name 'CGNet'

python test.py --gpu_id '1' --data_name 'WHU' --model_name 'CGNet'
'''

## Dataset Path Setting

Make sure your dataset follows this structure:
- LEVIR-CD
     |--train  
          |--A  (First temporal image)  
          |--B  (Second temporal image)  
          |--label (Ground truth)  
     |--val  
     |--test

