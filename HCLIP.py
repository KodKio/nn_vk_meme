import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import CLIPModel


class CLIPClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.use_pretrained_map = args.use_pretrained_map
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.weight_image_loss = args.weight_image_loss
        self.weight_text_loss = args.weight_text_loss

        self.acc = torchmetrics.Accuracy('binary')

        self.auroc = torchmetrics.AUROC('binary')
        self.precision_score = torchmetrics.Precision('binary')
        self.recall = torchmetrics.Recall('binary')
        self.f1 = torchmetrics.F1Score('binary')

        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        if self.use_pretrained_map:
            self.image_map = nn.Sequential(
                copy.deepcopy(self.clip.visual_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
            self.text_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )

        else:
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=0.2)]
            text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=0.2)]
            for _ in range(1, self.num_mapping_layers):
                image_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=0.2)])
                text_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=0.2)])

            self.image_map = nn.Sequential(*image_map_layers)
            self.text_map = nn.Sequential(*text_map_layers)

        pre_output_input_dim = self.map_dim
        pre_output_layers = [nn.Dropout(p=0.4)]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1:
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=0.1)])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers):
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=.1)])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)

        if self.weight_image_loss > 0:
            pre_output_layers = [nn.Dropout(p=0.4)]
            for _ in range(self.num_pre_output_layers):  # next pre-output layers
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=.1)])
            self.pre_output_image = nn.Sequential(*pre_output_layers)

            self.output_image = nn.Linear(output_input_dim, 1)

        if self.weight_text_loss > 0:
            pre_output_layers = [nn.Dropout(p=.4)]
            for _ in range(self.num_pre_output_layers):  # next pre-output layers
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=.1)])
            self.pre_output_text = nn.Sequential(*pre_output_layers)

            self.output_text = nn.Linear(output_input_dim, 1)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        del self.clip

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask']).pooler_output

        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1)  # [batch_size, d]

        features = torch.mul(image_features, text_features)  # [batch_size, d]

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch, batch_idx, calling_function='validation'):

        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        output = {}

        if self.weight_image_loss > 0:
            features_pre_output = self.pre_output_image(image_features)
            logits = self.output_image(features_pre_output).squeeze(dim=1)  # [batch_size, 1]
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()

            output['image_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output['image_accuracy'] = self.acc(preds, batch['labels'])
            output['image_auroc'] = self.auroc(preds_proxy, batch['labels'])

        if self.weight_text_loss > 0:
            features_pre_output = self.pre_output_text(text_features)
            logits = self.output_text(features_pre_output).squeeze(dim=1)  # [batch_size, 1]
            preds_proxy = torch.sigmoid(logits)
            preds = (preds_proxy >= 0.5).long()

            output['text_loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
            output['text_accuracy'] = self.acc(preds, batch['labels'])
            output['text_auroc'] = self.auroc(preds_proxy, batch['labels'])

        features = torch.mul(image_features, text_features)  # mul features

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1(or)n]
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='training')

        if self.weight_image_loss > 0:
            image_loss = output['image_loss']
        else:
            image_loss = 0

        if self.weight_text_loss > 0:
            text_loss = output['text_loss']
        else:
            text_loss = 0

        total_loss = output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])

        if self.weight_image_loss > 0:
            self.log('train/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log('train/text_loss', text_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx, calling_function='validation')

        if self.weight_image_loss > 0:
            image_loss = output['image_loss']
        else:
            image_loss = 0

        if self.weight_text_loss > 0:
            text_loss = output['text_loss']
        else:
            text_loss = 0

        total_loss = output['loss'] + self.weight_image_loss * image_loss + self.weight_text_loss * text_loss

        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'])
        self.log(f'val/auroc', output['auroc'])

        if self.weight_image_loss > 0:
            self.log(f'val/image_loss', image_loss)
        if self.weight_text_loss > 0:
            self.log(f'val/text_loss', text_loss)

        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        prefix_map = {
            0: 'dev_seen',
            1: 'test_seen',
        }
        prefix = prefix_map[dataloader_idx]
        if dataloader_idx == 0:
            calling_function = 'validation'
        elif dataloader_idx == 1:
            calling_function = 'training'

        output = self.common_step(batch, batch_idx, calling_function=calling_function)

        self.log(f'{prefix}/accuracy', output['accuracy'])
        self.log(f'{prefix}/auroc', output['auroc'])
        return output

    def on_training_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


def create_model(args):
    model = CLIPClassifier(args=args)
    return model
