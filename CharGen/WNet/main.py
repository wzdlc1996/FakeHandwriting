import torch
import itertools as itr

from torch.utils.data import DataLoader
from dataIO import ChineseCharDataset, TensorListToImage, TestCharDataset
from model import WNet, Discriminator, FeatClassifierForEncoders
from loss import LossG, LossD
from dataType import DiscriminatorOutput, GeneratorOutput, DataItem

dev = torch.device("cuda")

DataSet = ChineseCharDataset()
GenNet = WNet().to(dev)
DisNet = Discriminator(num_font=DataSet.getFontNumber(), num_char=DataSet.getCharNumber()).to(dev)

ThetaP = FeatClassifierForEncoders(DataSet.getCharNumber()).to(dev)
ThetaR = FeatClassifierForEncoders(DataSet.getFontNumber()).to(dev)

GenOptim = torch.optim.Adam(GenNet.parameters())
DisOptim = torch.optim.Adam(
    itr.chain(
        DisNet.parameters(),
        ThetaP.parameters(),
        ThetaR.parameters()
    )
)

GenLoss = LossG(dev)
DisLoss = LossD(DisNet, dev)


datald = DataLoader(
    dataset=DataSet
)

MAXEPOCH = 5000
sepr = 20

for epoch in range(MAXEPOCH):
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

    torch.save(GenNet.state_dict(), "./checkpoints/gen.ckpt")
    torch.save(DisNet.state_dict(), "./checkpoints/dis.ckpt")
    torch.save(ThetaR.state_dict(), "./checkpoints/ThetaR.ckpt")
    torch.save(ThetaP.state_dict(), "./checkpoints/ThetaP.ckpt")

    if epoch % int(MAXEPOCH / sepr) == 0:
        print(f"Complete epoch={epoch}, dis_loss={dis_loss_i}\tgen_loss={gen_loss_i}")
        res = []
        with torch.no_grad():
            tt = DataLoader(TestCharDataset())
            for proto, refer in tt:
                res.append(GenNet(proto.to(dev), refer.to(dev)).r.cpu()[0])

        TensorListToImage(res, "./results/res_%.2d.pdf" % epoch)








