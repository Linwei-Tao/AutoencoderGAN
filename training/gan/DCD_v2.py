import torch
import torch.nn.functional as F
import numpy as np

from third_party.gather_layer import GatherLayer
from training.criterion import nt_xent


def supcon_fake(out1, out2, others, temperature, distributed=False):
    if distributed:
        out1 = torch.cat(GatherLayer.apply(out1), dim=0)
        out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        others = torch.cat(GatherLayer.apply(others), dim=0)
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss

def debiased_log_softmax(sim_matrix, tau_plus, N, temperature):
    sim_real = sim_matrix[:,:2*N].fill_diagonal_(0)
    pos = torch.exp(sim_real)
    neg_real = torch.exp(sim_real).sum(1, keepdim=True) - pos - 1
    neg_fake = torch.exp(sim_matrix[:,2*N:]).sum(1, keepdim=True).repeat(1,2*N)
    Ng = (-tau_plus * (2*N-2+2*N) * pos + neg_real + neg_fake) / (1 - tau_plus)
    # constrain (optional)
    Ng = torch.clamp(Ng, min=(2*N-2+2*N) * np.e ** (-1 / temperature))
    lsm = torch.log(pos / (pos + Ng))/(2*N-1)
    return lsm.fill_diagonal_(0)

def DCD_loss(real1, real2, fake, temperature, params, distributed=False):
    if distributed:
        real1 = torch.cat(GatherLayer.apply(real1), dim=0)
        real2 = torch.cat(GatherLayer.apply(real2), dim=0)
        fake = torch.cat(GatherLayer.apply(fake), dim=0)
    N = real1.size(0)

    _out = [real1, real2, fake]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs[:2*N] @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    tau_plus = params.tau_plus
    M = params.M
    sim_matrix.fill_diagonal_(0)
    pos = torch.exp(sim_matrix[:, :2 * N])
    neg_real = torch.exp(sim_matrix[:, :2 * N]).sum(1, keepdim=True) - pos - 1
    neg_generate = torch.exp(sim_matrix[:, 2 * N:]).sum(1, keepdim=True).repeat(1, 2 * N)

    N_r = 2 * N - 2
    N_g = N
    N_rg = (-tau_plus * N_r * pos + neg_real) / (1 - tau_plus)
    # constrain
    N_rg = torch.clamp(N_rg, min=N_r * np.e ** (-1 / temperature))

    N_gg = (-tau_plus * N_g * pos + neg_generate) / (1 - tau_plus)
    # constrain
    N_gg = torch.clamp(N_gg, min=N_g * np.e ** (-1 / temperature))

    L_debiasedself = torch.log(pos / (pos + N_rg + N_gg)) / (2 * N - 1)
    L_debiasedself = L_debiasedself.fill_diagonal_(0)
    L_debiasedself = -L_debiasedself.sum(1)
    dcd_loss = L_debiasedself.mean()
    return dcd_loss

# DCD_v2 treat all neg_real as compensation M samples, corresponds to M positive views in Debiased Contrastive Learning
def DCD_loss_v2(real1, real2, fake1, fake2, temperature, params, distributed=False):
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

    tau_plus = params.tau_plus
    M = params.M
    sim_real = sim_matrix[:,:N].fill_diagonal_(0)
    pos = torch.exp(sim_real)
    neg_real = torch.exp(sim_real).sum(1, keepdim=True) - pos - 1
    Mpos = torch.exp(sim_real).mean(1, keepdim=True) - 1/N
    neg_fake = torch.exp(sim_matrix[:,N:]).sum(1, keepdim=True).repeat(1,N)
    Ng = (-tau_plus * (N-2+2*N) * Mpos + neg_real + neg_fake) / (1 - tau_plus)
    # constrain (optional)
    Ng = torch.clamp(Ng, min=(N-2+2*N) * np.e ** (-1 / temperature))
    lsm = torch.log(pos / (pos + Ng))/(N-1)
    lsm = lsm.fill_diagonal_(0)
    return -lsm.sum(1).mean()


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
    dcd_loss = DCD_loss_v2(real1, real2, fake1, fake2, temperature=P.temp, params=P, distributed=P.distributed)

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
