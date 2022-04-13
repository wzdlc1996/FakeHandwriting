import os
import torch
import itertools as itr
import hyperParam as hp

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataIO import ChineseCharDataset, TensorListToImage, TestCharDataset, EvalCharDataset
from model import WNet, Discriminator, FeatClassifierForEncoders
from loss import LossG, LossD
from dataType import DiscriminatorOutput, GeneratorOutput, DataItem

root_fold = "./results/res_0412"
dev = torch.device("cuda")

DataSet = ChineseCharDataset()
GenNet = WNet().to(dev)
DisNet = Discriminator(num_font=DataSet.getFontNumber(), num_char=DataSet.getCharNumber()).to(dev)

#  load check point
ckpt_folder = "results/res_0412/ckpt_00100"
GenNet.load_state_dict(torch.load(f"{ckpt_folder}/gen.ckpt"))
DisNet.load_state_dict(torch.load(f"{ckpt_folder}/dis.ckpt"))

res = []
with torch.no_grad():
    # tt = DataLoader(TestCharDataset(testPath="./chars/eval_8.txt"))
    tt = DataLoader(EvalCharDataset("./eval_hdwt/01.png", "我", "./chars/eval_8.txt"))
    for proto, refer in tt:
        res.append(GenNet(proto.to(dev), refer.to(dev)).r.cpu()[0])

TensorListToImage(res, "./eval_hdwt/generated.pdf")