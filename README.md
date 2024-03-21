# HW to Chapter 14 “Convolutional Network”

## Non-programming Assignment

### Q1. What is pooling layer and how it works?

Answer:  

A pooling layer in a convolutional neural network (CNN) serves to reduce the spatial dimensions (i.e., the width and height) of the input volume for the next convolutional layer. It is a form of non-linear down-sampling. There are several benefits to the pooling layer:

Reduction of Computation: By reducing the spatial size of the representation, the amount of parameters and computation in the network decreases, making the network more efficient.  

Control Overfitting: Reducing the spatial dimensions means there are fewer parameters in the network, which can help in reducing overfitting by providing an abstracted form of the representation.  

Spatial Variance: Pooling helps the network to be invariant to small translations of the input. A small shift in the input will not change the output of the pooling layer significantly, thereby making the network more robust to the position of features in the input image.

#### How Pooling Layer Works  

Pooling layers operate independently on each depth slice of the input and resize it spatially, using the MAX or AVERAGE operation:

Max Pooling: The most common form of pooling, where the maximum element from the portion of the image covered by the pooling window (e.g., a 2x2 window) is selected. It outputs the maximum value from the portion of the image covered by the kernel. For example, with a 2x2 pooling size, max pooling selects the maximum value out of the 4 values in the 2x2 sub-region.

Average Pooling: Computes the average of the elements in the portion of the image covered by the pooling window. Unlike max pooling, which focuses on the most prominent features, average pooling captures the average presence of features in the pooling window.

Global Pooling: Instead of operating on a small sub-region, global pooling operations summarize the entire depth slice into a single value. Global max pooling and global average pooling are two examples, where they take the maximum or average of an entire depth slice, respectively.

#### Operational Details  

A pooling layer typically operates on each input channel independently, preserving the depth of the input volume.
The pooling operation involves sliding a window (or kernel) across each channel of the input image. For each location the window covers, it computes the maximum (for max pooling) or average (for average pooling) value of that region and outputs it to the next layer.
Pooling layers commonly use a stride of more than one to reduce the spatial dimensions of the output volume significantly. For instance, a 2x2 pooling filter applied with a stride of 2 reduces the size of the input by half in both dimensions.
In summary, the pooling layer simplifies the information in the input volume, making the representation smaller and more manageable for the network to process, while retaining essential information about the presence of features.

### Q2. What are three major types of layers in the convolutional neural network?

Answer:

In a Convolutional Neural Network (CNN), there are three major types of layers that play distinct roles in processing and learning from input data:

1. Convolutional Layer  

Function: The primary purpose of the convolutional layer is to extract features from the input image. This is achieved through the application of various filters (also known as kernels) that scan across the image and perform convolution operations. Each filter is designed to detect specific types of features, such as edges, textures, or patterns.  

Output: The result of this layer is a set of feature maps that represent the presence of specific features detected at various locations in the input image. As the network deepens, convolutional layers can extract more complex and high-level features.  

2. Pooling (Subsampling or Down-sampling) Layer  

Function: The pooling layer follows the convolutional layer and is used to reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer. It helps to decrease the computational power required to process the data through dimensionality reduction. Additionally, it is useful for extracting dominant features which are rotational and positional invariant, thus providing the network with more abstracted features.  

Types: The most common types of pooling are max pooling and average pooling. Max pooling returns the maximum value from the portion of the image covered by the pool, while average pooling returns the average of all values in that portion.  

Output: A reduced feature map that emphasizes the most prominent features, helping to make the feature detection process more efficient and less sensitive to the exact location of features in the input image.  

3. Fully Connected (FC) Layer  

Function: Fully connected layers, sometimes simply called dense layers, are layers where every neuron in the layer is connected to every neuron in the preceding layer. The main purpose of a fully connected layer is to use these connections to classify the input image into various classes based on the high-level features extracted by the convolutional and pooling layers.  

Output: Outputs from fully connected layers are typically the final classifications scores. In a multi-class classification problem, for example, each neuron in the final fully connected layer represents a specific class, and the value of the neuron represents the probability that the input image belongs to that class.  

Role in CNNs: While convolutional and pooling layers are responsible for feature extraction and abstraction, fully connected layers serve as classifiers based on those features.  

Summary:  

Together, these layers enable CNNs to perform complex image recognition and classification tasks. Convolutional layers and pooling layers handle feature extraction and spatial hierarchy, while fully connected layers make decisions (e.g., classification) based on those features. This structured approach allows CNNs to learn patterns in visual data with remarkable efficiency and accuracy.

#### Q3. What is the architecture of a convolutional network?  

Answer:  

The architecture of a Convolutional Neural Network (CNN) is designed to automatically and adaptively learn spatial hierarchies of features from input images for tasks such as image classification, object detection, and many others. A CNN architecture typically consists of several layers that process and transform the input image into outputs such as class scores or encoded features for complex tasks. Here is a general outline of a typical CNN architecture, segmented into its core components:

1. Input Layer  

Purpose: Serves as the entry point for the input image data into the network. The input layer holds the raw pixel values of the image, which are typically normalized before being processed by subsequent layers.  

Format: The input is usually a three-dimensional array with dimensions corresponding to image height, image width, and color channels (e.g., RGB).  

2. Convolutional Layers  

Purpose: These layers are the core building blocks of a CNN. Convolutional layers apply a set of learnable filters to the input. Each filter is spatially small but extends through the full depth of the input volume. As the filter slides (or convolves) across the input image, it produces a two-dimensional activation map that gives the responses of that filter at every spatial position.  

Output: The collection of these activation maps forms the output volume of the convolutional layer, providing a set of filtered images that highlight specific types of features (e.g., edges, textures).  

3. Activation Function  

ReLU (Rectified Linear Unit): After each convolution operation, an activation function like ReLU is applied to introduce non-linear properties to the system, allowing the network to solve more complex problems. ReLU is commonly used because it accelerates the convergence of stochastic gradient descent compared to sigmoid or tanh functions.  

4. Pooling (Subsampling) Layers  

Purpose: Pooling layers reduce the dimensions (width and height, not depth) of the input volume for the next convolutional layer. This is done to decrease the computational power required, control overfitting by providing an abstracted form of the representation, and reduce the sensitivity to the exact location of features.  

Common Types: Max pooling (taking the maximum element from the portion of the image covered by the pooling kernel) and average pooling (taking the average of all elements in that portion).  

5. Fully Connected (FC) Layers  

Purpose: After several convolutional and pooling layers, the high-level reasoning in the network is done by fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular neural networks. Their role is typically to output the scores corresponding to different classes based on the features extracted by the convolutional and pooling layers.  

Classification: In a classification task, the final fully connected layer will have as many neurons as there are classes, with the output of each neuron representing the score or probability of the input belonging to a particular class.  

6. Output Layer  

Purpose: The last fully connected layer is often followed by a softmax activation function in classification tasks, which converts scores into probability distributions over predicted classes.  

Summary of a Typical CNN Architecture Flow:

Input Image -> [Convolutional Layer + Activation Function] -> Pooling Layer -> [Convolutional Layer + Activation Function] -> Pooling Layer -> ... -> Fully Connected Layer -> Output  

Additional Components  

Batch Normalization: Often applied after convolutional layers but before the activation function to normalize the output of the previous layer; improves stability and performance.  

Dropout: Used as a regularization technique, where randomly selected neurons are ignored during training to prevent overfitting.  

This structure allows CNNs to capture the spatial and temporal dependencies in an image through the application of relevant filters, making them exceptionally well-suited for image recognition tasks
