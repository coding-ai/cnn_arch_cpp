# C++ Deep Neural Network Architectures

This repository contains two simple exmaples of how to replicate famous CNN architectures (LeNet and AlexNet) and try them out on the MNIST dataset.

## LeNet5 archietcture

Proposed by Yann LeCun, Leon Bottou, Youshua Bengio and Patrick Haffner in the 1990's, LeNet-5 is has become one of the most popular CNN architectures in the world. It was designed for the handwritten character recognition.

The architecture consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier, see image [here](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg).

The architecture is describe in the following lines of the code:

```
using net_type = loss_multiclass_log<
                                fc<10,        
                                relu<fc<84,   
                                relu<fc<120,  
                                max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                input<matrix<unsigned char>> 
                                >>>>>>>>>>>>;
```

To test the architecture I have attached a pre-built executable, run the following command inside the `lenet5` folder:

`./lenet5 ../data`

## AlexNet architecture

The proposed architecture was built for the ImageNet Large-Scale Visual Recgonition Challenge (ILSVRC) and consists of five convolutional layers and three fully-connected layers, using ReLU to speed up the computation and dropout to avoid overfitting, see the image [here](https://miro.medium.com/max/1400/1*qyc21qM0oxWEuRaj-XJKcw.png).

The architecture is describe in the following lines of the code:

```
using net_type = loss_multiclass_log<
                                fc<10,        
                                dropout<relu<fc<4096,   
                                dropout<relu<fc<4096, 
                                max_pool<3, 3, 2, 2, relu<con<256, 3, 3, 1, 1,  
                                relu<con<384, 3, 3, 1, 1, 
                                relu<con<384, 3, 3, 1, 1,
                                max_pool<3, 3, 2, 2, l2normalize<relu<con<256, 5, 5, 1, 1, 
                                max_pool<3, 3, 2, 2, l2normalize<relu<con<96, 11, 11, 1, 1, 
                                input<matrix<unsigned char>>>>>>>>>>>>>>>>>>>>>>>>>;
```

To test the architecture (on MNIST dataset) I have attached a pre-built executable, run the following command inside the `alexnet` folder:

`./alexnet ../data`

## Other architectures

You can easily replicate architectures by changing the `net_type` argument and building the solution. You can also try different combinations of learning rate for example or build your own architecture. A good way to start learning about Convolutional Neural Networks!
