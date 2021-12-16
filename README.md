# VRDL HW3 Instance Segmentation
This is the code for VRDL HW2. I used yolov4 as the architecture to do the homework. I trained my model with darknet first, then used the pytorch version of yolov4 to inference the test dataset. here are my reference:
- [U-Net](https://github.com/bvezilic/Nuclei-segmentation)

## dependencies
- python 3.8+
- pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1
- skimage
- tqdm
- matplotlib
- numpy
- PIL
- scipy
- cocoapi

## How to reproduce my result
1. download my code
2. setup dependencies
3. download [weightfile](https://drive.google.com/file/d/1UFBeXCQXDWSldn5Jwq3ScKstC3s2JQ4y/view?usp=sharing)
4. put the weightfile under the directory
5. put the test imges under ./dataset in local
6. run inference.py to reproduce my result
