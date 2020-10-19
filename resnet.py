import io
import numpy as np
import os
from torch import nn
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime
import tensorflow as tf
from onnx_tf.backend import prepare
from PIL import Image
from torchvision import transforms
import time

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
# device = 'cpu'
# Talk about
# 1. tracing vs. scripting
# https://pytorch.org/docs/stable/onnx.

# 2. Use of external data format
# 3. Training vs. Evaluation mode
def export_to_onnx_pt(model, batch, use_dynamic=True, use_external=False):
    input_names = ["input_1"]
    output_names = ["output1"]
    if use_dynamic:
        torch_out = torch.onnx.export(model,  # model being run
                                      batch,  # model input (or a tuple for multiple inputs)
                                      "models/resnet_dynamic.onnx",
                                      # where to save the model (can be a file or file-like object)
                                      input_names=input_names,
                                      output_names=output_names,
                                      export_params=True,
                                      use_external_data_format=use_external,
                                      dynamic_axes={'input_1': [0]},
                                      training=torch.onnx.TrainingMode.EVAL)
    else:
        torch_out = torch.onnx.export(model,  # model being run
                                      batch,  # model input (or a tuple for multiple inputs)
                                      "models/resnet.onnx",  # where to save the model (can be a file or file-like object)
                                      input_names=input_names,
                                      output_names=output_names,
                                      export_params=True,
                                      use_external_data_format=use_external,
                                      training=torch.onnx.TrainingMode.EVAL)

    # The resulting resnet.onnx is a binary protobuf file which contains both the network
    # structure and parameters of the model you exported
    # The keyword argument verbose=True causes the exporter to print out a human-readable representation of the network:
    # onnx_model = onnx.load("resnet.onnx")

    # Check that the IR is well formed
    # onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    # onnx.helper.printable_graph(onnx_model.graph)


def load_data(img_name):
    img = Image.open(img_name)
    return img


def prepare_data_pt(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.repeat(64, 1, 1, 1)
    return batch_t


def prepare_data_tf(img):
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.convert_image_dtype(image_array, tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.central_crop(img, 224 / 256)
    img = img / 255.0
    offset = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3])
    img -= offset
    scale = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3])
    img /= scale
    batch_t = tf.expand_dims(img, axis=0)
    batch_t = tf.repeat(batch_t, repeats=[64], axis=0)
    # The batch shape is (N, H, W, C) and Tensorflow uses shape (N, C, H, W).
    batch_t = tf.transpose(batch_t, [0, 3, 1, 2])
    return batch_t


def onnx_to_tf(onnx_model):
    tf_model = prepare(onnx_model)
    return tf_model


# Pytorch lacks support for converting onnx back to a Pytorch graph!
# See: https://github.com/plstcharles/thelper/pull/20
def onnx_to_pt(onnx_model):
    pass


# Load pre-trained resnet101 model
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
# Save to a Pytorch specific format
# torch.save(resnet.state_dict(), 'resnet.pt')
resnet.eval()

img = load_data("dog.jpg")
# Convert image into Pytorch and Tensorflow batch
# batch_tf = prepare_data_tf(img)
batch_pt = prepare_data_pt(img)

# Export to ONNX format
export_to_onnx_pt(resnet, batch_pt, use_dynamic=False, use_external=False)
export_to_onnx_pt(resnet, batch_pt, use_dynamic=True, use_external=False)

# prime the data transfer:
torch.rand((12, 3, 224, 224)).to(device)

# Run resnet101 on the batch and record execution time
start = time.time()
resnet.to(device)
print('Model data transfer time: {0}'.format(time.time() - start))
out_1 = resnet(batch_pt.to(device))
# out_2 = torch.nn.functional.softmax(out_1[0], dim=0)
print('Resnet-101 (regular) execution: {0}'.format(time.time() - start))


#####################################
# Convert back onnx to pytorch and run pytorch model on the data
# implementation lacking..
# pt_model = onnx_to_pt(onnx_model)
#####################################

#####################################
# Convert onnx to Tensorflow and run tensorflow model on the data
# tf_model = onnx_to_tf(onnx_model)
start = time.time()
# output = tf_model.run(batch_tf)
print('Resnet-101 Tensorflow execution (B=64): {0}'.format(time.time() - start))
#####################################

################################################################
# Now use ONNX Runtime
ep_list = onnxruntime.get_available_providers()
sess_options = onnxruntime.SessionOptions()
ep_dev_map = {'CUDAExecutionProvider': "cuda", 'CPUExecutionProvider': "cpu"}
output_dir = "models"
for i in range(0,2):
    for ep in ep_list:
        dev = ep_dev_map.get(ep)
        dynamic = {"non_dynamic": "models/resnet.onnx", "dynamic": "models/resnet_dynamic.onnx"}
        for k, v in dynamic.items():

            # This will save the optimized graph to the directory specified in optimized_model_filepath
            sess_options.optimized_model_filepath = os.path.join(output_dir, "resnet_optimized_model_{}.onnx".format(dev))
            ort_session = onnxruntime.InferenceSession(v, sess_options)
            # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
            output_name = ort_session.get_outputs()[0].name
            # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
            input_name = ort_session.get_inputs()[0].name

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            ort_inputs = {input_name: to_numpy(batch_pt)}
            ort_session.set_providers([ep]) # ep: Execution Provider - eg., CUDAExecutionProvider, CPUExecutionProvider
            # compute ONNX Runtime output prediction
            start = time.time()
            ort_outs = ort_session.run([output_name], ort_inputs)
            print("Resnet-101 OnnxRuntime {}, {} Inference time = {} ms".format(dev, k, time.time() - start))


