import os
import torch
import itertools as itr
import hyperParam as hp

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from dataIO import ChineseCharDataset, TensorListToImage, TestCharDataset
from model import WNet, Discriminator, FeatClassifierForEncoders
from loss import LossG, LossD
from dataType import DiscriminatorOutput, GeneratorOutput, DataItem

root_fold = "./results/res_0412"
dev = torch.device("cuda")

DataSet = ChineseCharDataset()
GenNet = WNet().to(dev)
DisNet = Discriminator(num_font=DataSet.getFontNumber(), num_char=DataSet.getCharNumber()).to(dev)

ThetaP = FeatClassifierForEncoders(DataSet.getCharNumber()).to(dev)
ThetaR = FeatClassifierForEncoders(DataSet.getFontNumber()).to(dev)

#  load check point
ckpt_folder = "results/res_0410/ckpt_00080"
GenNet.load_state_dict(torch.load(f"{ckpt_folder}/gen.ckpt"))
DisNet.load_state_dict(torch.load(f"{ckpt_folder}/dis.ckpt"))
ThetaP.load_state_dict(torch.load(f"{ckpt_folder}/ThetaP.ckpt"))
ThetaR.load_state_dict(torch.load(f"{ckpt_folder}/ThetaR.ckpt"))

GenOptim = torch.optim.AdamW(GenNet.parameters(), betas=hp.adamBeta, lr=hp.iniLr)
DisOptim = torch.optim.AdamW(
    itr.chain(
        DisNet.parameters(),
        # ThetaP.parameters(),
        # ThetaR.parameters()
    ),
    betas=hp.adamBeta,
    lr=hp.iniLr
)
GenLRSch = ExponentialLR(GenOptim, gamma=hp.lrDecay)
DisLRSch = ExponentialLR(DisOptim, gamma=hp.lrDecay)

GenLoss = LossG(dev)
DisLoss = LossD(DisNet, dev)


datald = DataLoader(
    dataset=DataSet,
    batch_size=hp.batch_size,
    shuffle=True,
    num_workers=hp.num_workers
)  # add batch_size > 1 would lead bug for loss._calGradientPenalty

MAXEPOCH = hp.MAXEPOCH
sepr = hp.sepr
dis_loss_i = gen_loss_i = 0.
for epoch in range(MAXEPOCH + 1):
    for datai in datald:
        datai: DataItem = datai.to(dev)
        ### train disciminator
        DisNet.zero_grad()
        gen_fake: GeneratorOutput = GenNet(datai.prototype_img, datai.reference_img)
        enc_p = ThetaP(gen_fake.lout.view(-1, 512).detach())
        enc_r = ThetaR(gen_fake.rout.view(-1, 512).detach())
        dis_fake: DiscriminatorOutput = DisNet(gen_fake.r, datai.prototype_img, datai.reference_img)
        dis_real: DiscriminatorOutput = DisNet(datai.real_img, datai.prototype_img, datai.reference_img)

        dis_loss = DisLoss(dis_fake, dis_real, gen_fake, enc_p, enc_r, datai)
        dis_loss_i = dis_loss.item()

        dis_loss.backward()
        DisOptim.step()

        ### train generator
        GenNet.zero_grad()
        gen_fake: GeneratorOutput = GenNet(datai.prototype_img, datai.reference_img)
        gen_byfk: GeneratorOutput = GenNet(gen_fake.r, gen_fake.r)
        dis_fake: DiscriminatorOutput = DisNet(gen_fake.r, datai.prototype_img, datai.reference_img)
        dis_real: DiscriminatorOutput = DisNet(datai.real_img, datai.prototype_img, datai.reference_img)
        enc_p = ThetaP(gen_fake.lout.view(-1, 512).detach())
        enc_r = ThetaR(gen_fake.rout.view(-1, 512).detach())

        gen_loss = GenLoss(dis_fake, dis_real, gen_fake, gen_byfk, enc_p, enc_r, datai)
        gen_loss_i = gen_loss.item()
        gen_loss.backward()
        GenOptim.step()

    #  update the learning rate
    GenLRSch.step()
    DisLRSch.step()

    if epoch % int(MAXEPOCH / sepr) == 0:
        ckpt_fold = f"{root_fold}/ckpt_%.5d" % epoch
        os.mkdir(ckpt_fold)
        torch.save(GenNet.state_dict(), f"{ckpt_fold}/gen.ckpt")
        torch.save(DisNet.state_dict(), f"{ckpt_fold}/dis.ckpt")
        torch.save(ThetaR.state_dict(), f"{ckpt_fold}/ThetaR.ckpt")
        torch.save(ThetaP.state_dict(), f"{ckpt_fold}/ThetaP.ckpt")

        print(f"Complete epoch={epoch}, dis_loss=%.4f\tgen_loss=%.4f" % (dis_loss_i, gen_loss_i))
        res = []
        with torch.no_grad():
            tt = DataLoader(TestCharDataset())
            for proto, refer in tt:
                res.append(GenNet(proto.to(dev), refer.to(dev)).r.cpu()[0])

        TensorListToImage(res, f"{root_fold}/res_%.5d.pdf" % epoch)








