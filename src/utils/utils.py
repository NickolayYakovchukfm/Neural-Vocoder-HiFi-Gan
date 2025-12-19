from hydra.utils import instantiate


def get_losses(config, device):
    return {
        loss_name: instantiate(config.loss[loss_name]).to(device)
        for loss_name in config.loss
    }


def get_optimizers(config, model):
    g_params = model.Generator.parameters()
    d_params = list(model.MultiScaleDiscriminator.parameters()) + list(
        model.MultiPeriodDiscriminator.parameters()
    )
    return {
        "g_optimizer": instantiate(config.optimizer, params=g_params),
        "d_optimizer": instantiate(config.optimizer, params=d_params),
    }


def get_lr_schedulers(config, optimizers):
    return {
        "g_lr_scheduler": instantiate(
            config.lr_scheduler, optimizer=optimizers["g_optimizer"]
        ),
        "d_lr_scheduler": instantiate(
            config.lr_scheduler, optimizer=optimizers["d_optimizer"]
        ),
    }