import torch
import torch.nn.functional as F
import numpy as np

from third_party.gather_layer import GatherLayer
from training.criterion import nt_xent


# separate tau_plus for neg_real and neg_generated
# 2N for generated N for real, based on DCD(the first version)
# based on Contrastive Learning with Hard Negative Samples
def DCD_loss_v6(real1, real2, fake1, fake2, temperature, params, distributed=False):
    if distributed:
        real1 = torch.cat(GatherLayer.apply(real1), dim=0)
        real2 = torch.cat(GatherLayer.apply(real2), dim=0)
        fake1 = torch.cat(GatherLayer.apply(fake1), dim=0)
        fake2 = torch.cat(GatherLayer.apply(fake2), dim=0)
    N = real1.size(0)

    _out = [real1, fake1, fake2]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)
    sim_matrix = sim_matrix[:N]

    tau_plus4gen = params.tau_plus4gen
    tau_plus = params.tau_plus
    M = params.M
    sim_matrix.fill_diagonal_(0)
    pos = torch.exp(sim_matrix[:, :N])
    neg_real = torch.exp(sim_matrix[:, :N]).sum(1, keepdim=True) - pos - 1
    neg_generate = torch.exp(sim_matrix[:, N:]).sum(1, keepdim=True).repeat(1, N)


    N_r = N - 2
    N_g = 2 * N

    imp = (params.beta4r * neg_real.log()).exp()
    reweight_neg_real = (imp * neg_real).sum(dim=-1) / imp.mean(dim=-1)
    N_rg = (-tau_plus * N_r * pos + reweight_neg_real) / (1 - tau_plus)
    # constrain
    N_rg = torch.clamp(N_rg, min=N_r * np.e ** (-1 / temperature))

    imp = (params.beta4g * neg_generate.log()).exp()
    reweight_neg_generate = (imp * neg_generate).sum(dim=-1) / imp.mean(dim=-1)
    N_gg = (-tau_plus4gen * N_g * pos + reweight_neg_generate) / (1 - tau_plus4gen)
    # constrain
    N_gg = torch.clamp(N_gg, min=N_g * np.e ** (-1 / temperature))


    L_debiasedself = torch.log(pos / (pos + N_rg + N_gg)) / (N - 1)
    L_debiasedself = L_debiasedself.fill_diagonal_(0)
    L_debiasedself = -L_debiasedself.sum(1)
    dcd_loss = L_debiasedself.mean()
    return dcd_loss





def loss_D_fn(P, D, options, images, gen_images):
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)

    cat_images = torch.cat([images, images, gen_images, gen_images], dim=0)
    d_all, aux = D(P.augment_fn(cat_images), sg_linear=True, projection=True, projection2=True)

    # dcd_loss
    samples = aux['projection2']
    samples = F.normalize(samples)
    real1, real2, fake1, fake2 = samples[:N], samples[N:2*N], samples[2*N:3*N], samples[3*N:]
    dcd_loss = DCD_loss_v6(real1, real2, fake1, fake2, temperature=P.temp, params=P, distributed=P.distributed)

    # dis loss
    d_real, d_gen = d_all[:N], d_all[2*N:3*N]
    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    elif options['loss'] == 'lsgan':
        d_loss_real = ((d_real - 1.0) ** 2).mean()
        d_loss_fake = (d_gen ** 2).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    else:
        raise NotImplementedError()

    return dcd_loss, {
        "penalty": d_loss,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }


def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss
