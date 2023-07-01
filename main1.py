import scipy
import numpy as np
import time
import imageio
from skimage.transform import resize
import cv2

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std,
                                       size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):
        self.input = input
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding + self.input.shape[2],
        self.padding:self.padding + self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        # self.output[idxn, idxc, idxh, idxw] = ____________
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.input_pad[idxn, :, idxh * self.stride:idxh * self.stride + self.kernel_size,
                            idxw * self.stride:idxw * self.stride + self.kernel_size] * self.weight[:, :, :, idxc]) + \
                                                              self.bias[idxc]

        return self.output

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        start_time = time.time()
        self.input = input
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算最大池化层的前向传播，取池化窗口内的最大值
                        # self.output[idxn.idxc, idxh, idxw] = _______
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc, idxh * self.stride:idxh * self.stride + self.kernel_size,
                            idxw * self.stride:idxw * self.stride + self.kernel_size])

        return self.output


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = top_diff
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input):
        self.input = input
        output = np.maximum(self.input, 0)
        return output

    def backward(self, top_diff):
        bottom_diff = np.multiply(np.sign(np.maximum(self.input, 0)), top_diff)
        return bottom_diff


class SoftmaxLossLayer(object):
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        # TODO：softmax 损失层的前向传播，计算输出结果
        partsum = np.sum(input_exp, axis=1, keepdims=True)
        sum = np.tile(partsum, (10, 1))
        self.prob = input_exp / partsum.T
        return self.prob

    def ger_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        # bottom_diff = (self.prob - self.label_onehot)
        return bottom_diff


class VGG19(object):
    def __init__(self, param_path='imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )

    def load_image(self, image_dir):
        print('loading and preprocessing image from ' + image_dir)
        # self.input_image = scipy.misc.imread(image_dir)
        self.input_image = imageio.v2.imread(image_dir)

        # self.input_image = scipy.misc.imresize(self.input_image, [244, 244, 3])
        self.input_image = resize(self.input_image, output_shape=[244, 244, 3])
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))

        # input dim[N, channel, height, width]
        self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])



    def build_model(self):
        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)

        self.layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(3, 128, 128, 1, 1)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2)

        self.layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_4'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2)

        self.layers['conv4_1'] = ConvolutionalLayer(3, 256, 512, 1, 1)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_4'] = ReLULayer()
        self.layers['pool4'] = MaxPoolingLayer(2, 2)

        self.layers['conv5_1'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_3'] = ReLULayer()
        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(2, 2)

        self.layers['flatten'] = FlattenLayer([512, 7, 7], [512 * 7 * 7])
        self.layers['fc6'] = FullyConnectedLayer(512 * 7 * 7, 4096)
        self.layers['relu6'] = ReLULayer()

        self.layers['fc7'] = FullyConnectedLayer(4096, 4096)
        self.layers['relu7'] = ReLULayer()

        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)
        self.layers['softmax'] = SoftmaxLossLayer()
        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        for idx in range(43):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)
            if idx >= 37 and 'fc' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx - 1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    def forward(self):
        current = self.input_image
        for idx in range(len(self.param_layer_name)):
            current = self.layers[self.param_layer_name[idx]].forward(current)
        return current

    def evaluate(self):
        prob = self.forward()
        top1 = np.argmax(prob[0])
        print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image('2.jpg')
    vgg.evaluate()
