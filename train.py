import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
import argparse
from models import *
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime

from dataset_G import *
from mindspore import context
import mindspore.nn as nn
from mindspore import nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms
from mindspore import Model


parser = argparse.ArgumentParser(description='MindSpore FDSR')
parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
#context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")



parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--parameter',  default='./para', help='name of parameter file')
parser.add_argument('--model',  default='DSR_Net', help='choose model')
parser.add_argument('--lr',  default='0.0005', type=float, help='learning rate')
parser.add_argument('--result',  default='./result', help='learning rate')
parser.add_argument('--epoch',  default=1000, type=int, help='max epoch')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-s_%s'%(opt.result, s, opt.lr, opt.scale)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)
net = Net(num_feats=32, depth_chanels=1, color_channel=3, kernel_size=3)
net.set_train()

criterion = nn.L1Loss()

optimizer = nn.Adam(net.trainable_params(), learning_rate=opt.lr, weight_decay=0.0)
print('===> Loading datasets')

nyu_dataset = dataset_G(root_dir='/opt/data/private/depthsr/hc/data/nyuv2_npy/')

rescale = 1.0 / 255.0
shift = 0.0
num_parallel_workers = 1
hwc2chw_op = CV.HWC2CHW()
type_cast_op = C.TypeCast(mstype.float32)

def dataset_info(dt):
    assert dt in ['EORSSD']
    if dt == 'EORSSD':
        dt_mean = [0.3412, 0.3798, 0.3583]
        dt_std = [0.1148, 0.1042, 0.0990]
    return dt_mean, dt_std
    
dt_mean, dt_std = dataset_info('EORSSD')
trans_totensor = Compose([py_vision.ToTensor()])
trans_totensor_rgb = Compose([py_vision.ToTensor(), py_vision.Normalize(dt_mean, dt_std)])

train_dataset = ds.GeneratorDataset(source=nyu_dataset, column_names=['guidance', 'target', 'gt'],
                                        num_parallel_workers=num_parallel_workers,
                                        shuffle=False).map(
        operations=type_cast_op, input_columns=['guidance'], num_parallel_workers=num_parallel_workers).map(
        operations=type_cast_op, input_columns=['target'], num_parallel_workers=num_parallel_workers).map(
        operations=type_cast_op, input_columns=['gt'], num_parallel_workers=num_parallel_workers).map(
        operations=trans_totensor_rgb, input_columns="guidance", num_parallel_workers=num_parallel_workers).map(
        operations=trans_totensor, input_columns="target", num_parallel_workers=num_parallel_workers).map(
        operations=trans_totensor, input_columns="gt", num_parallel_workers=num_parallel_workers)

train_dataset = train_dataset.batch(1, True)        
max_epoch = opt.epoch

class TrainOneStepCell(nn.Cell):
    def __init__(self, network):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.dsr_weights = ParameterTuple(network.dsr_subnet.trainable_params())
        self.dsr_optimizer = network.dsr_optimizer

        self.dsr_subnet = nn.WithLossCell(network.dsr_subnet, network.dsr_criterion)
        self.dsr_grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.dsr_sens = dsr_sens

    def construct(self, input, input_rgb, target):
        dsr_weights = self.dsr_weights

        dsr_loss = self.network.backward(input, input_rgb, target)

        dsr_sens = ops.Fill()(ops.DType()(dsr_loss), ops.Shape()(dsr_loss), self.dsr_sens)
        dsr_grads = self.dsr_grad(self.dsr_subnet, dsr_weights)(input, target, dsr_sens)


        return dsr_loss, self.dsr_optimizer(dsr_grads)

train_net = TrainOneStepCell(model)

t = tqdm(iter(train_dataset), leave=True, total=len(nyu_dataset))
for idx, data in enumerate(t):
    guidance, target, gt = data[0], data[1], data[2]
    t0 = time.time()
    dsr_loss, dsr_grad= train_net(guidance, target, gt)
    t1 = time.time()
    dsr_loss = dsr_loss.asnumpy()
    epoch_loss_dsr += dsr_loss
    iteration += 1
    print(
            "===> Epoch[{}]({}/{}): SR Loss: {:.4f} || Timer: {:.4f} sec.".format(
                epoch, iteration, training_data_loader.get_dataset_size(), dsr_loss, (t1 - t0)))
print(
        "===> Epoch {} Complete: Avg. SR Loss: {:.4f} || Avg. MDE Loss: {:.4f}.".format(epoch,
                                                                                        epoch_loss_dsr / training_data_loader.get_dataset_size(), time.ctime())

    
ckptconfig = CheckpointConfig(keep_checkpoint_max=1000)
model_out_path = result_root
ckpoint_cb = ModelCheckpoint(prefix='111',directory=model_out_path, config=ckptconfig)
out = Model(net, loss_fn=criterion, optimizer=optimizer)
out.train(1000, train_dataset, callbacks= ckpoint_cb)
mox.file.copy(model_out_path+ str(epoch) + '.ckpt')
print("Checkpoint saved to {}".format(model_out_path))


def validate(net, root_dir='/opt/data/private/depthsr/hc/data/nyuv2_npy/'):

    nyu_test = dataset_G(root_dir='/opt/data/private/depthsr/hc/data/nyuv2_npy/')
    train_dataset = ds.GeneratorDataset(source=nyu_test, column_names=['guidance', 'target', 'gt'],
                                        num_parallel_workers=num_parallel_workers,
                                        shuffle=True).map(
        operations=type_cast_op, input_columns=['guidance'], num_parallel_workers=num_parallel_workers).map(
        operations=type_cast_op, input_columns=['target'], num_parallel_workers=num_parallel_workers).map(
        operations=type_cast_op, input_columns=['gt'], num_parallel_workers=num_parallel_workers).map(
        operations=rescale_op, input_columns=['iguidance'], num_parallel_workers=num_parallel_workers).map(
        operations=rescale_op, input_columns=['target'], num_parallel_workers=num_parallel_workers).map(
        operations=rescale_op, input_columns=['gt'], num_parallel_workers=num_parallel_workers).map(
        operations=hwc2chw_op, input_columns=['guidance'], num_parallel_workers=num_parallel_workers).map(
        operations=hwc2chw_op, input_columns=['target'], num_parallel_workers=num_parallel_workers).map(
        operations=hwc2chw_op, input_columns=['gt'], num_parallel_workers=num_parallel_workers)

    
    net.eval()
    rmse = np.zeros(449)
    test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        minmax = test_minmax[:,idx]
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        out = net((guidance, target))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(), minmax)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse
    

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[1]-minmax[0]) + minmax[1]
    b = b*(minmax[1]-minmax[0]) + minmax[1]
    
    return np.sqrt(np.mean(np.power(a-b,2)))

def print_network(net):
    num_params = 0
    for param in net.trainable_params():
        num_params += np.prod(param.shape)
    print('Total number of parameters: %d' % num_params)

