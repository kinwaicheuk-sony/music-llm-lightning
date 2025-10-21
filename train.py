import argparse, os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from mllm.common_utils.config_utils import instantiate_from_config
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    config = OmegaConf.load(args.config)
    os.makedirs(f"{config.lightning.logdir}/{config.project}/{config.lightning.version_name}", exist_ok=True)
    os.system(f"cp {args.config} {config.lightning.logdir}/{config.project}/{config.lightning.version_name}/config.yaml")
    pl.seed_everything(42)
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()
    model = instantiate_from_config(config.model)
    trainer_config = config.lightning.trainer
    if trainer_config.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        raise ValueError(f"Invalid strategy: {trainer_config.strategy}")
    del trainer_config["strategy"]
    trainer_kwargs = vars(argparse.Namespace(**trainer_config))
    logger = pl.loggers.TensorBoardLogger(
        save_dir=config.lightning.logdir,
        name=config.project,
        version=config.lightning.version_name,
    )
    callbacks = []
    for cb_name, cb_conf in config.lightning.callbacks.items():
        if cb_name in ["save_checkpoint", "learning_rate"]:
            callback = instantiate_from_config(cb_conf)
            callbacks.append(callback)
    trainer = pl.Trainer(
        callbacks=callbacks,
        strategy=strategy,
        devices="auto",
        logger=logger,
        **trainer_kwargs
    )
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume if args.resume else None
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpu", default=None, type=str, required=False, help="GPU ID")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main(args)
