# tensor-flow-with-Cnn-or-Dnn
CNN and DNN using tenser flow to predict that men is suffer from cancer or not. 

Convolutional layer:
a)Convolution is a mathematical operation that’s used in single processing to filter signals,
 find patterns in signals.
b)In a convolutional layer, all neurons apply convolution operation to the inputs,
hence they are called convolutional neurons
c)The most important parameter in a convolutional neuron is the filter size.
Pooling layer:
Pooling layer is mostly used immediately after the convolutional layer to
reduce the spatial size(only width and height, not depth).
b)This reduces the number of parameters, hence computation is reduced.
c)The most common form of pooling is Max pooling where we take a filter of size F*F
 and apply the maximum operation over the F*F sized part of the image.
 Creating Network layer:
 Building convolution layer in TensorFlow:
 tf.nn.conv2d function can be used to build a convolutional layer which takes these inputs:
 a)Input= the output(activation) from the previous layer.
 b)Filter= trainable variables defining the filter. We start with a random normal distribution and learn these weights
 b)Strides= defines how much you move your filter when doing convolution.
 In this function, it needs to be a Tensor of size>=4.
 c)Padding=SAME means we shall 0 pad the input such a way that output x,y
 dimensions are same as that of input.
 Flattening layer:
 The Output of a convolutional layer is a multi-dimensional Tensor.
  We want to convert this into a one-dimensional tensor. This is done in the Flattening layer
 Placeholders and input:
 let’s create a placeholder that will hold the input training images.
  All the input images are read in dataset.py file and resized to 128 x 128 x 3 size
 Network design:
  We use the functions defined above to create various layers of the network.
 Predictions:
 probability of each class by applying softmax to the output of fully connected layer.
By using CNN we firstly trained te data and overload the data  by using activation function 
like softmax .

We got the loss fuction:.02
Accuracy(prediction): 84%

By using DNN classifier 

we predict the data on basis of features:


feature_names = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
                 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses']

