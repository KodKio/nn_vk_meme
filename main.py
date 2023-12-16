from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from HCLIP import create_model
from dataset import load_dataset, CustomCollator

from HCLIP import CLIPClassifier


class Args:
    pass


args = Args()
args.clip_pretrained_model = "openai/clip-vit-large-patch14"  # "openai/clip-vit-base-patch32"
args.use_pretrained_map = False
args.image_size = 224
args.num_mapping_layers = 1
args.map_dim = 768
args.num_pre_output_layers = 1
args.freeze_image_encoder = True
args.freeze_text_encoder = True
args.limit_train_batches = 1.0
args.limit_val_batches = 1.0
args.max_epochs = 20
args.log_every_n_steps = 50
args.batch_size = 16
args.val_check_interval = 1.0
args.lr = 1e-4
args.weight_image_loss = 1.0
args.weight_text_loss = 1.0
args.weight_decay = 1e-4
args.gradient_clip_val = 0.1


#model = CLIPClassifier.load_from_checkpoint(checkpoint_path='checkpoints/elated-waterfall-2-epoch=12.ckpt',args=args, strict=False)
#print(model)

def main(args):
    dataset_train = load_dataset(args=args, split='train')
    dataset_val = load_dataset(args=args, split='dev')

    collator = CustomCollator(args)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, collate_fn=collator)

    model = create_model(args)

    monitor = "val/auroc"
    project = "meme"

    wandb_logger = WandbLogger(project=project, config=args)
    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    wandb_logger.experiment.config.update(num_params)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename=wandb_logger.experiment.name + '-{epoch:02d}',
                                          monitor=monitor, mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)
    trainer = Trainer(accelerator="gpu", max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=wandb_logger, log_every_n_steps=args.log_every_n_steps,
                      val_check_interval=args.val_check_interval,
                      callbacks=[checkpoint_callback],
                      limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
                      deterministic=True)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(ckpt_path='best', dataloaders=[dataloader_val])
