from importlib import import_module


def setup(P):
    mod = import_module(f'.{P.mode}', 'training.gan')
    loss_G_fn = mod.loss_G_fn
    loss_D_fn = mod.loss_D_fn

    if P.mode == 'std':
        filename = f"{P.mode}_{P.penalty}"
        if 'cr' in P.penalty:
            filename += f'_{P.aug}'
    elif P.mode == 'aug':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'aug_both':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'simclr_only':
        filename = f"{P.mode}_{P.aug}_T{P.temp}"
    elif P.mode == 'contrad':
        filename = f"{P.mode}_{P.aug}_L{P.lbd_a}_T{P.temp}"
    elif P.mode == 'debiased_contrad_v2':
        filename = f"{P.mode}_{P.aug}_Tau{P.tau_plus}_M{P.M}"
    elif P.mode == 'DCD':
        filename = f"{P.mode}_{P.aug}_tau{P.tau_plus}_M{P.M}"
    elif P.mode == 'DCD+simclr':
        filename = f"{P.mode}_{P.aug}_tau{P.tau_plus}_M{P.M}"
    elif P.mode == 'contrad+autoencoder':
        filename = f"{P.mode}_{P.aug}_MSE{P.mse_factor}_DIS{P.dis_genral_factor}"
    elif P.mode == 'contrad+autoencoder2':
        filename = f"{P.mode}_{P.aug}_MSE_Img{P.mse_factor_on_img}_MSE_z{P.mse_factor_on_z}_DIS{P.dis_genral_factor}_Z_Constrain{P.z_constrain_factor}"
    elif P.mode == "sphereContraD":
        filename = f"{P.mode}_DViewConstrain{P.D_view_constrain}_DDis{P.D_Dis}_GZConstrain{P.G_z_constrain}"
    else:
        filename = f"{P.mode}_{P.aug}_tau{P.tau_plus}_tau4gen{P.tau_plus4gen}_M{P.M}_beta4r{P.beta4r}_beta4g{P.beta4g}"

    P.filename = filename
    P.train_fn = {
        "G": loss_G_fn,
        "D": loss_D_fn
    }
    return P