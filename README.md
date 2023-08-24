
# A Framework for Pareto Multi Task Learning with Completed Scalarization Functions
This is the official implementation for [Paper]([paper](https://arxiv.org/abs/2302.12487))  

<img src="src/1.jpg" alt=”Image” style="width:1200px;height:400px;">

Support MOO experiments in `MOP/` with problems:
- 2D: ex1, ex2, ex4, ZDT1-4
- 3D: ex3, DTLZ2

ex3                   |  DTLZ2
:-------------------------:|:-------------------------:
![](src/train_1.gif)  |  ![](src/train_2.gif)

Support MTL experiments in `MTL/experiments/Multi_task/` and `MTL/experiments/Multi_output/` with datasets:
- Multi-MNIST/Multi-Fashion/Multi-Fash+MNIST 
- CelebA 
- NYUv2 
- SARCOS

Multi-MNIST                   |  Multi-Fashion                 |  Multi-Fash+MNIST 
:-------------------------:|:-----------------------:|:-------------------------:
![](src/MNIST.jpg)  |  ![](src/FASHIOn.jpg) | ![](src/Fashion_Mnist.jpg)

## Requirements
- Pytorch
- Python
- Latex

## Installation
```
pip3 install -r requirements.txt
```
## Contact

[]([trananhtuan23012000@gmail.com](https://github.com/tuantran23012000))  
*Please create an issue or contact me through trananhtuan23012000@gmail.com, thanks!*

## Author

Tran Anh Tuan

## Citation
If our framework is useful for your research, please consider to cite the paper:
```
@article{tran2023framework,
  title={A framework for controllable pareto front learning with completed scalarization functions and its applications},
  author={Tran, Tuan Anh and Hoang, Long Phi and Le, Dung Duy and Tran, Thang Ngoc},
  journal={arXiv preprint arXiv:2302.12487},
  year={2023}
}
```
