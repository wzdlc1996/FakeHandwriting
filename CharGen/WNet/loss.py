from impts import *
from hyperParam import *

from torch import Tensor
from model import Discriminator
from dataType import DataItem, DiscriminatorOutput, GeneratorOutput


class _rmseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, yhat):
        return torch.sqrt(self.mse(yhat, y))


def _calGradientPenalty(
        D: Discriminator,
        x_real: Tensor,
        x_fake: Tensor,
        x_proto: Tensor,
        x_refer: Tensor,
        device: torch.device
) -> Tensor:
    """
    Calculate the gradient penalty for WGAN.

    :param D: (Discriminator) defined by model
    :param x_real: real instance
    :param x_fake: fake instance generated by G
    :param x_proto: prototype of the x, for Discriminator input
    :param x_refer: reference of the x, for Discriminator input
    :return:
    """
    interpol_factor = torch.rand(x_real.shape[0], 1, 1, 1).to(device)
    interpol_img = interpol_factor * x_real + (1 - interpol_factor) * x_fake

    interpol_out: DiscriminatorOutput = D(interpol_img, x_proto, x_refer)
    interpol_img.require_grad = True

    # print(interpol_out.r_gan.size())
    # exit(0)
    grad = torch.autograd.grad(
        outputs=interpol_out.r_gan,
        inputs=interpol_img,
        grad_outputs=[torch.ones(interpol_out.r_gan.size(), device=device)],
        create_graph=True
    )

    grad_norm = 0
    for gr in grad:
        grad_norm += (gr - torch.ones(gr.size(), device=device)).pow(2).mean()
    grad_norm = grad_norm.sqrt()
    return grad_norm


class LossG(nn.Module):
    """
    The loss for generator, by the paper, it is
        L = ( - alpha * L_adv
        + beta_d * L_dac
        + beta_p * L_epc
        + beta_r * L_erc
        + lamb_l1 * L_1
        + lamb_phi * L_phi
        + psi_p * L_cp
        + psi_r * L_cr
        )
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(
            self,
            d_out_by_G: DiscriminatorOutput,
            d_out_by_Real: DiscriminatorOutput,
            g_out: GeneratorOutput,
            g_out_by_G: GeneratorOutput,
            enc_p: Tensor,
            enc_r: Tensor,
            target: DataItem
    ) -> Tensor:
        """


        During the training setup, We need two outputs by D and two outputs by G:
            1.  d_out_by_G: D(G(x_j^0, x_k^i))
            2.  d_out_by_Real: D(x_j^i)
            3.  g_out: G(x_j^0, x_k^i)
            4.  g_out_by_G: G(g_out, g_out), mainly for Enc_p(g_out) and Enc_r(g_out)

        Apart from these, we need additional two fully-connecting soft-max classifier for encoder outputs
            1.  enc_p: C_encp(Enc_p(x_j^0)) ~ j (of prototype char)
            2.  enc_r: C_encr(Enc_r(x_k^i)) ~ i (of styled font)

        :param d_out_by_G: (DiscriminatorOutput), the output of Disriminator from Generator
        :param d_out_by_Real: (DiscriminatorOutput), the output of Disriminator from Real data
        :param g_out: (GeneratorOutput), the output of Generator: G(x_proto, x_refer)
        :param g_out_by_G: (GeneratorOutput), the output of Generator: G(x_gen, x_gen) where x_gen = G(x_proto, x_refer)
        :param enc_p: (Tensor), the output by additional fully connecting softmax classifier:
            theta_p(x_gen.lout.view(-1, 512))
        :param enc_r: (Tensor), the output by additional fully connecting softmax classifier:
            theta_r(x_gen.lout.view(-1, 512))
        :param target: (DataItem), by query from dataloader
        :return:
        """

        #  GAN loss
        r_gan_sz = d_out_by_G.r_gan.size()
        lab = torch.full(r_gan_sz, real_lab, dtype=torch.float).to(self.device)
        # l_adv = nn.BCELoss()(d_out_by_G.r_gan, lab)
        l_adv = d_out_by_G.r_gan

        #  Categorical loss of discriminator auxiliary classifier
        l_dac = nn.CrossEntropyLoss()(d_out_by_Real.r_font, target.reference_ind) \
            + nn.CrossEntropyLoss()(d_out_by_G.r_font, target.reference_ind)

        #  Reconstruction loss
        l_1 = nn.L1Loss()(g_out.r, target.real_img)
        #  in paper it is VGG-16 net, here use conv_features instead for simplicity
        l_phi = torch.sqrt(
            torch.sum(
                #  Use stack instead `cat` for 0-d tensors by MSELoss() returns
                torch.stack([nn.MSELoss()(y, yhat) for y, yhat in zip(d_out_by_Real.feat, d_out_by_G.feat)], dim=0),
                dim=0
            )
        )

        #  Constant loss of encoders
        l_cp = nn.MSELoss()(g_out.lout, g_out_by_G.lout)
        l_cr = nn.MSELoss()(g_out.rout, g_out_by_G.rout)

        # Categorical loss on both encoders
        l_epc = nn.CrossEntropyLoss()(enc_p, target.character_ind)
        l_erc = nn.CrossEntropyLoss()(enc_r, target.reference_ind)

        return (
            alpha * l_adv + beta_d * l_dac + beta_p * l_epc + beta_r * l_erc
            + lamb_l1 * l_1 + lamb_phi * l_phi + psi_p * l_cp + psi_r * l_cr
        ).mean()


class LossD(nn.Module):
    def __init__(self, D: Discriminator, device):
        super().__init__()
        self.disc = D
        self.device = device

    def forward(
            self,
            d_out_by_G: DiscriminatorOutput,
            d_out_by_Real: DiscriminatorOutput,
            g_out: GeneratorOutput,
            enc_p: torch.Tensor,
            enc_r: torch.Tensor,
            target: DataItem
    ) -> Tensor:
        """


        During the training setup, We need two outputs by D and one output by G:
            1.  d_out_by_G: D(G(x_j^0, x_k^i))
            2.  d_out_by_Real: D(x_j^i)
            3.  g_out: G(x_j^0, x_k^i)

        Apart from these, we need additional two fully-connecting soft-max classifier for encoder outputs
            1.  enc_p: C_encp(Enc_p(x_j^0)) ~ j (of prototype char)
            2.  enc_r: C_encr(Enc_r(x_k^i)) ~ i (of styled font)

        :param d_out_by_G: (DiscriminatorOutput), the output of Disriminator from Generator
        :param d_out_by_Real: (DiscriminatorOutput), the output of Disriminator from Real data
        :param g_out: (GeneratorOutput), the output of Generator: G(x_proto, x_refer)
        :param g_out_by_G: (GeneratorOutput), the output of Generator: G(x_gen, x_gen) where x_gen = G(x_proto, x_refer)
        :param enc_p: (Tensor), the output by additional fully connecting softmax classifier:
            theta_p(x_gen.lout.view(-1, 512))
        :param enc_r: (Tensor), the output by additional fully connecting softmax classifier:
            theta_r(x_gen.lout.view(-1, 512))
        :param target: (DataItem), by query from dataloader
        :return:
        """
        #  GAN loss
        r_gan_sz = d_out_by_G.r_gan.size()
        rlab = torch.full(r_gan_sz, real_lab, dtype=torch.float).to(self.device)
        flab = torch.full(r_gan_sz, fake_lab, dtype=torch.float).to(self.device)
        # l_adv = nn.BCELoss()(d_out_by_G.r_gan, flab) + nn.BCELoss()(d_out_by_Real.r_gan, rlab)
        l_adv = d_out_by_Real.r_gan - d_out_by_G.r_gan

        # gradient penalty
        l_gp = _calGradientPenalty(
            self.disc,
            target.real_img,
            g_out.r,
            target.prototype_img,
            target.reference_img,
            self.device
        )

        #  Categorical loss of discriminator auxiliary classifier
        l_dac = nn.CrossEntropyLoss()(d_out_by_Real.r_font, target.reference_ind) \
            + nn.CrossEntropyLoss()(d_out_by_G.r_font, target.reference_ind)

        # Categorical loss on both encoders
        l_epc = nn.CrossEntropyLoss()(enc_p, target.character_ind)
        l_erc = nn.CrossEntropyLoss()(enc_r, target.reference_ind)

        return (alpha * l_adv + alpha_GP * l_gp + beta_d * l_dac + beta_p * l_epc + beta_r * l_erc).mean()


if __name__ == "__main__":
    from model import WNet, Discriminator, FeatClassifierForEncoders
    from dataIO import ChineseCharDataset

    gen = WNet()
    dis = Discriminator(25, 140)

    dev = torch.device("cuda:0")

    gl = LossG(dev)
    dl = LossD(dis, dev)

    theta_p = FeatClassifierForEncoders(140)
    theta_r = FeatClassifierForEncoders(25)

    datal = torch.utils.data.DataLoader(ChineseCharDataset())
    targ: DataItem = next(iter(datal))

    x = targ.real_img
    xg: GeneratorOutput = gen(targ.prototype_img, targ.reference_img)
    gg: GeneratorOutput = gen(xg.r, xg.r)
    d_real: DiscriminatorOutput = dis(targ.real_img, targ.prototype_img, targ.reference_img)
    d_g: DiscriminatorOutput = dis(xg.r, targ.prototype_img, targ.reference_img)
    print(gl(d_g, d_real, xg, gg, theta_p(xg.lout.view(-1, 512)), theta_r(xg.rout.view(-1, 512)), targ))
    print(dl(d_g, d_real, xg, theta_p(xg.lout.view(-1, 512)), theta_r(xg.rout.view(-1, 512)), targ))

