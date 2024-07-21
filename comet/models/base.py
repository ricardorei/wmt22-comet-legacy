# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
CometModel
========================
    Abstract Model class that implements some of the Pytorch Lightning logic.
    Extend this class to create new model and metrics within COMET.
"""
import abc
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
from xmlrpc.client import boolean

import numpy as np
import pytorch_lightning as ptl
import torch
import transformers
from comet.encoders import str2encoder
from comet.modules import LayerwiseAttention
from comet.modules.losses import KLLoss, MSELoss
from packaging import version
from torch.utils.data import DataLoader, RandomSampler, Subset

from .lru_cache import tensor_lru_cache
from .pooling_utils import average_pooling, max_pooling
from .predict_pbar import PredictProgressBar
from .utils import DataSampler, OrderedSampler, Prediction, Target

if "COMET_EMBEDDINGS_CACHE" in os.environ:
    CACHE_SIZE = int(os.environ["COMET_EMBEDDINGS_CACHE"])
else:
    CACHE_SIZE = 1024


logger = logging.getLogger(__name__)


class CometModel(ptl.LightningModule, metaclass=abc.ABCMeta):
    """CometModel:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    :param class_identifier: subclass identifier.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        layer_transformation: str = "softmax",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        sampling_gamma: float = 1.0,
        validation_data: Optional[List[str]] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["load_weights_from_checkpoint"])

        if self.hparams.encoder_model == "XLM-RoBERTa-XL":
            # Ensure backwards compatibility with transformer versions
            if version.parse(transformers.__version__) < version.parse("4.17.0"):
                raise Exception(
                    "XLM-RoBERTa-XL requires transformers>=4.17.0. Your current version is {}".format(
                        transformers.__version__
                    )
                )

        self.encoder = str2encoder[self.hparams.encoder_model].from_pretrained(
            self.hparams.pretrained_model
        )

        self.epoch_nr = 0
        if self.hparams.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=self.hparams.layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=True,
            )
        else:
            self.layerwise_attention = None

        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            logger.info("Encoder model fine-tuning")
            self._frozen = False

        if self.hparams.keep_embeddings_frozen:
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs
        self.mc_dropout = False  # Flag used to control usage of MC Dropout
        self.caching = False  # Flag used to control Embedding Caching
        self.initialize_loss()

    def set_mc_dropout(self, value: bool):
        self.mc_dropout = value

    @abc.abstractmethod
    def read_training_data(self):
        pass

    @abc.abstractmethod
    def read_validation_data(self):
        pass

    @abc.abstractmethod
    def prepare_sample(
        self,
        sample: List[Dict[str, Union[str, float]]],
        stage: str = "fit",
        # words: boolean = True,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def init_metrics(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def is_referenceless(self) -> bool:
        pass

    def freeze_encoder(self) -> None:
        logger.info("Encoder model frozen.")
        self.encoder.freeze()

    def initialize_loss(self) -> None:
        if self.hparams.loss == "kl":
            self.loss = KLLoss()
        elif self.hparams.loss == "mse":
            self.loss = MSELoss()
        else:
            raise Exception(
                "{} is not a valid loss function.".format(self.hparams.loss)
            )

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        if self.hparams.loss == "kl":
            return self.loss(prediction.score, prediction.std, target.score, target.std)
        return self.loss(prediction.score, target.score)

    def unfreeze_encoder(self) -> None:
        if self._frozen:
            if self.trainer.is_global_zero:
                logger.info("Encoder model fine-tuning")

            self.encoder.unfreeze()
            self._frozen = False
            if self.hparams.keep_embeddings_frozen:
                self.encoder.freeze_embeddings()

    def on_train_epoch_end(self) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if self.epoch_nr >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    def set_embedding_cache(self):
        """Function that when called turns embedding caching on."""
        self.caching = True

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_word_embds: bool = False,
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
            a single sentence.

        :param tokens: sequences [batch_size x seq_len]
        :param lengths: lengths [batch_size]

        :return: torch.Tensor [batch_size x hidden_size]
        """
        if self.caching:
            return self.retrieve_sentence_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_word_embds=return_word_embds,
            )
        else:
            return self.compute_sentence_embedding(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
                return_word_embds=return_word_embds,
            )

    @tensor_lru_cache(maxsize=CACHE_SIZE)
    def retrieve_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_word_embds: bool = False,
    ) -> torch.Tensor:
        """Wrapper for `get_sentence_embedding` function that caches results."""
        return self.compute_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_word_embds=return_word_embds,
        )

    def compute_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_word_embds: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        if self.layerwise_attention:
            # HACK: LayerNorm is applied at the MiniBatch. This means that for big batch sizes the variance
            # and norm within the batch will create small differences in the final score
            # If we are predicting we split the data into equal size batches to minimize this variance.
            if not self.training:
                n_splits = len(torch.split(encoder_out["all_layers"][-1], 8))
                embeddings = []
                for split in range(n_splits):
                    all_layers = []
                    for layer in range(len(encoder_out["all_layers"])):
                        layer_embs = torch.split(encoder_out["all_layers"][layer], 8)
                        all_layers.append(layer_embs[split])
                    split_attn = torch.split(attention_mask, 8)[split]
                    embeddings.append(self.layerwise_attention(all_layers, split_attn))
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = self.layerwise_attention(
                    encoder_out["all_layers"], attention_mask
                )

        elif self.hparams.layer >= 0 and self.hparams.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.hparams.layer]

        else:
            raise Exception("Invalid model layer {}.".format(self.hparams.layer))

        if self.hparams.pool == "default":
            sentemb = encoder_out["sentemb"]

        elif self.hparams.pool == "max":
            sentemb = max_pooling(
                input_ids, embeddings, self.encoder.tokenizer.pad_token_id
            )

        elif self.hparams.pool == "avg":
            sentemb = average_pooling(
                input_ids,
                embeddings,
                attention_mask,
                self.encoder.tokenizer.pad_token_id,
            )

        elif self.hparams.pool == "cls":
            sentemb = embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        if return_word_embds:
            # TODO: try
            #  return sentemb, embeddings
            return sentemb, encoder_out["wordemb"]
        else:
            return sentemb

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> torch.Tensor:
        """
        Runs one training step and logs the training loss.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: Loss value
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """
        Runs one validation step and logs metrics.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        self.log("val_loss", loss_value, on_step=True, on_epoch=True)
        if dataloader_idx == 0:
            self.train_metrics.update(batch_prediction.score, batch_target["score"])
        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                batch_prediction.score, batch_target["score"]
            )

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        if self.mc_dropout:
            self.train()
        else:
            self.eval()

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs one prediction step and returns the predicted values.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        if self.mc_dropout:
            outputs = [self(**batch) for _ in range(self.mc_dropout)]
            mcd_output = Prediction(
                **{k: torch.stack([dic[k] for dic in outputs]) for k in outputs[0]}
            )
            mcd_output["std"] = mcd_output.scores.std(dim=0)
            mcd_output["mcd_scores"] = mcd_output.scores.T
            mcd_output["scores"] = mcd_output.scores.mean(dim=0)
            return mcd_output

        return self(**batch)

    def validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            results = self.val_metrics[i].compute()
            self.val_metrics[i].reset()
            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()}, prog_bar=True
        )

    def setup(self, stage) -> None:
        """Data preparation function called before training by Lightning.

        :param stage: either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in (None, "fit"):
            if self.hparams.sampling_gamma < 1.0:
                train_dataset = DataSampler(
                    self.hparams.train_data,
                    self.read_training_data,
                    gamma=self.hparams.sampling_gamma,
                )
            else:
                train_dataset = self.read_training_data(self.hparams.train_data[0])

            self.validation_sets = [
                self.read_validation_data(d) for d in self.hparams.validation_data
            ]

            self.first_epoch_total_steps = len(train_dataset) // (
                self.hparams.batch_size * max(1, self.trainer.num_devices)
            )
            # Always validate the model with part of training.
            train_subset = np.random.choice(
                a=len(train_dataset), size=min(1000, len(train_dataset) * 0.2)
            )
            self.train_subset = Subset(train_dataset, train_subset)
            self.init_metrics()

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        if self.hparams.sampling_gamma < 1.0:
            train_dataset = DataSampler(
                self.hparams.train_data,
                self.read_training_data,
                gamma=self.hparams.sampling_gamma,
            )
        else:
            data_path = self.hparams.train_data[
                self.current_epoch % len(self.hparams.train_data)
            ]
            train_dataset = self.read_training_data(data_path)
            logger.info(f"Loading {data_path}.")

        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        val_data = [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                num_workers=2 * self.trainer.num_devices,
            )
        ]
        for validation_set in self.validation_sets:
            val_data.append(
                DataLoader(
                    dataset=validation_set,
                    batch_size=self.hparams.batch_size,
                    collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                    num_workers=2 * self.trainer.num_devices,
                )
            )
        return val_data

    def prepare_for_inference(self, sample):
        """Ideally this should be a lamba function but for some reason python does not copy local lambda functions.
        This functions replaces `collate_fn=lambda x: self.prepare_sample(x, inference=True)` from line 434.
        """
        return self.prepare_sample(sample, stage="predict")

    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 8,
        gpus: int = 1,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "ddp",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Union[Tuple[List[float], float], Tuple[List[float], List[float], float]]:
        """Function that receives a list of samples (dictionaries with translations, sources and/or references)
        and returns segment level scores and a system level score. If `mc_dropout` is set, it also returns for each
        segment score, a confidence value.

        :param samples: List with dictionaries with source, translations and/or references.
        :param batch_size: Batch size used during inference.
        :param gpus: Number of GPUs to be used.
        :param mc_dropout: Number of inference steps to run using MCD. Its disabled by default!
        :param progress_bar: Flag that turns on and off the predict progress bar.
        :param accelarator: Pytorch Lightning accelerator (e.g: dp, ddp).
        :param num_workers: Number of workers to use when loading data from dataloaders.
        :param length_batching: If set to true, reduces padding by sorting samples by MT length.

        :return: List with segment-level scores and a system-score or segment-level scores, segment-level
            confidence and a system-score.
        """

        def restore_list_order(sorted_list, sort_ids):
            """Restores the original ids of a given list."""
            unsorted_list = [None for _ in range(len(sorted_list))]
            for i, s in zip(sort_ids, sorted_list):
                unsorted_list[i] = s
            return unsorted_list

        def flatten_predictions(predictions):
            predictions = Prediction(**{k: [dic[k] for dic in predictions] for k in predictions[0]})
            for k, v in predictions.items():
                if torch.is_tensor(v[0]):
                    # If we have tensors we can use cat to flatten them.
                    predictions[k] = torch.cat(v, dim=0).tolist()
                else:
                    # for other predictions such as word tags we have to flatten the list.
                    predictions[k] = [item for sublist in v for item in sublist]
            return predictions
        
        # HACK: Workaround pytorch bug that prevents ParameterList to be used in DP
        # https://github.com/pytorch/pytorch/issues/36035
        if self.layerwise_attention is not None and gpus > 1:
            self.layerwise_attention.gamma_value = float(
                self.layerwise_attention.gamma[0]
            )
            self.layerwise_attention.weights = [
                float(parameter[0])
                for parameter in self.layerwise_attention.scalar_parameters
            ]

        # TODO: ideally this should be based on the actual token_ids
        # but that would require fundamentally changing the way dataloader is
        # setup, so currently raw chars are used as an approximation
        sampler = None
        if length_batching and gpus < 2:
            try:
                sort_ids = np.argsort([len(sample["src"]) for sample in samples])
            except KeyError:
                sort_ids = np.argsort([len(sample["ref"]) for sample in samples])
            sampler = OrderedSampler(sort_ids)

        if num_workers is None:
            num_workers = 2 * gpus

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
        )
        accelerator = accelerator if gpus > 1 else None

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        if progress_bar:
            trainer = ptl.Trainer(
                gpus=gpus,
                deterministic=True,
                logger=False,
                callbacks=[PredictProgressBar()],
                accelerator=accelerator,
                max_epochs=-1,
            )
        else:
            trainer = ptl.Trainer(
                gpus=gpus,
                deterministic=True,
                logger=False,
                progress_bar_refresh_rate=0,
                accelerator=accelerator,
                max_epochs=-1,
            )

        # TODO:
        # Remove this upon resolution of:
        # https://github.com/PyTorchLightning/pytorch-lightning/discussions/11392
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Your `predict_dataloader`'s sampler has shuffling enabled.*",
        )

        if mc_dropout > 0:
            self.set_mc_dropout(mc_dropout)

        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=True
        )
        predictions = flatten_predictions(predictions)
        if length_batching and gpus < 2:
            predictions = Prediction(**{k: restore_list_order(v, sort_ids) for k, v in predictions.items()})
            
        predictions["system_score"] = sum(predictions.score) / len(predictions.score)
        return predictions
