from models.yolo import Model
from utils.torch_utils import ModelEMA
import torch
cfg = 'models/yolov5s_car_sign.yaml'
nc = 3
model = Model(cfg, ch=3, nc=nc).to(torch.device('cuda:0'))
ema = ModelEMA(model)
ckpt = {'model': ema.ema}
torch.save(ckpt, 'yolov5s_nofocus_relu.pt')
