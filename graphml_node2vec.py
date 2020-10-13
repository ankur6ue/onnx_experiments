import os.path as osp
import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import onnxruntime
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                 context_size=10, walks_per_node=10, num_negative_samples=1,
                 sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

def export_to_onnx_pt(model, data, use_dynamic=True):
    input_names = ["input_1"]
    output_names = ["output1"]
    batch = torch.arange(data.num_nodes)
    if use_dynamic:
        torch_out = torch.onnx.export(model,  # model being run
                                      batch,  # model input (or a tuple for multiple inputs)
                                      "models/graphml/node2vec.onnx",
                                      # where to save the model (can be a file or file-like object)
                                      input_names=input_names,
                                      output_names=output_names,
                                      export_params=True,
                                      use_external_data_format=False,
                                      dynamic_axes={'input_1': [0]},
                                      training=torch.onnx.TrainingMode.EVAL)
    else:
        torch_out = torch.onnx.export(model,  # model being run
                                      batch,  # model input (or a tuple for multiple inputs)
                                      "models/graphml/node2vec.onnx",
                                      # where to save the model (can be a file or file-like object)
                                      input_names=input_names,
                                      output_names=output_names,
                                      export_params=True,
                                      use_external_data_format=False,
                                      training=torch.onnx.TrainingMode.EVAL)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc

model_path = './models/graphml/node2vec.pkl'
if not os.path.exists(model_path):
    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    fl = open(model_path, 'wb')
    pickle.dump(model, fl)
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        export_to_onnx_pt(model, data, use_dynamic=True)

@torch.no_grad()
def plot_points(colors):
    model.eval()
    start = time.time()
    batch = torch.arange(data.num_nodes, device=device)
    z = model(batch)
    print('Node2Vec execution time: {0}'.format(time.time() - start))

    # Now use onnx
    sess_options = onnxruntime.SessionOptions()
    # This will save the optimized graph to the directory specified in optimized_model_filepath
    sess_options.optimized_model_filepath = os.path.join("./models/graphml",
                                                         "node2vec_optimized_model_{}.onnx".format(device))
    ort_session = onnxruntime.InferenceSession("models/graphml/node2vec.onnx", sess_options)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction

    # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
    output_name = ort_session.get_outputs()[0].name

    # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: to_numpy(batch)}

    ort_session.set_providers(['CPUExecutionProvider'])

    # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
    output_name = ort_session.get_outputs()[0].name
    start = time.time()
    ort_outs = ort_session.run([output_name], ort_inputs)
    print('Node2Vec (ONNX) execution: {0}'.format(time.time() - start))

    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
plot_points(colors)