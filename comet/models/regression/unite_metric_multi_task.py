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
UniTE Metric
========================
    Implementation of the UniTE metric proposed in 
    [UniTE: Unified Translation Evaluation](https://arxiv.org/pdf/2204.13346.pdf)
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn

from comet.models.pooling_utils import max_pooling, average_pooling
from comet.models.regression.regression_metric import RegressionMetric
from comet.models.regression.referenceless import ReferencelessRegression
from comet.models.utils import Prediction, Target
from comet.modules import FeedForward, LayerwiseAttention
from comet.models.word_level_utils import convert_word_tags
from comet.modules.losses import KLLoss, MSELoss
from transformers.optimization import Adafactor


class UniTEMetricMT(ReferencelessRegression):
    """UniTEMetric:

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
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param final_activation: Feed Forward final activation.
    :param input_segments: Input sequences used during training/inference.
        ["mt", "src"] for QE, ["mt", "ref"] for reference-base evaluation and ["mt", "src", "ref"]
        for full sequence evaluation.
    :param unite_training: If set to true the model is trained with UniTE loss that combines QE
        with Metrics.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
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
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "cls",
        sent_layer: Union[str, int] = "mix",
        word_layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        sampling_gamma: float = 1.0,
        hidden_sizes: List[int] = [2304, 768],
        rnn_hidden_size: int = 1024,
        rnn_hidden_layers: int = 2,
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_segments: Optional[List[str]] = ["mt", "src", "ref"],
        load_weights_from_checkpoint: Optional[str] = None,
        unite_training: Optional[bool] = False,
        qe_training: Optional[bool] = False,
        word_level_training: Optional[bool] = False,
        word_weights: List[float] = [0.5, 0.5],
        loss_lambda: Optional[float] = 0.5,
        use_rnn: Optional[bool] = True,
    ) -> None:
        super(RegressionMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            pool=pool,
            layer=sent_layer,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            sampling_gamma=sampling_gamma,
            validation_data=validation_data,
            load_weights_from_checkpoint=load_weights_from_checkpoint,
            class_identifier="unite_metric_multi_task",
        )
        self.save_hyperparameters(ignore=["load_weights_from_checkpoint"])
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=self.loss.target_dim,
        )

        if self.hparams.use_rnn:
            # rnn approach
            self.word_scorer_rnn = nn.LSTM(
                input_size=self.encoder.output_units,
                hidden_size=self.hparams.rnn_hidden_size,
                num_layers=self.hparams.rnn_hidden_layers,
                batch_first=True,
                dropout=self.hparams.dropout,
                bidirectional=True,
            )

        self.hidden2tag = nn.Linear(
            2 * self.hparams.rnn_hidden_size if self.hparams.use_rnn else self.encoder.output_units, 
            2
        )  # CZ-TODO: remove hardcoding of num_classes for wordlevel

        if self.hparams.sent_layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=True,
            )
        else:
            self.layerwise_attention = None

        if self.hparams.word_layer == "mix":
            self.word_layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=True,
            )
        else:
            self.word_layerwise_attention = None

        self.input_segments = input_segments
        self.unite_training = unite_training
        self.qe_training = qe_training
        self.word_level_training = word_level_training
        # self.encoder.add_special_eos_token("<EOS>")

    def is_referenceless(self) -> bool:
        return True

    def set_input_segments(self, input_segments: List[str]):
        assert input_segments in [
            ["mt", "src"],
            ["mt", "ref"],
            ["mt", "src", "ref"],
        ], (
            "Input segments is ['mt', 'src'] for QE, ['mt', 'ref'] for reference-based evaluation"
            "and ['mt', 'src', 'ref'] for complete sequence evaluation."
        )
        self.input_segments = input_segments

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]

        word_level_parameters = []
        if self.hparams.use_rnn:
            word_level_parameters.append({
                "params": self.word_scorer_rnn.parameters(),
                "lr": self.hparams.learning_rate,
            })
        word_level_parameters.append({"params": self.hidden2tag.parameters(), "lr": self.hparams.learning_rate})

        layerwise_attn_params = []
        if self.layerwise_attention:
            layerwise_attn_params.append({
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                })

        if self.word_layerwise_attention:
            layerwise_attn_params.append({
                "params": self.word_layerwise_attention.parameters(),
                "lr": self.hparams.learning_rate
            })

        if self.layerwise_attention or self.word_layerwise_attention:
            params = (
                layer_parameters
                + top_layers_parameters
                + word_level_parameters
                + layerwise_attn_params
            )
        else:
            params = layer_parameters + top_layers_parameters + word_level_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        return [optimizer], []

    def read_training_data(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)

        if self.loss.target_dim == 2:
            df = df[self.input_segments + ["score", "std"]]
            df["std"] = df["std"].astype("float16")
        elif self.word_level_training:
            df = df[self.input_segments + ["score"] + ["mt_tags"]]
        else:
            df = df[self.input_segments + ["score"]]

        for segment in self.input_segments:
            df[segment] = df[segment].astype(str)

        if self.word_level_training:
            df["mt_tags"] = convert_word_tags(df["mt_tags"].to_list())

        df["score"] = df["score"].astype("float16")
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Reads a comma separated value file for validation.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        if self.word_level_training:
            df = df[self.input_segments + ["score"] + ["mt_tags"]]
        else:
            df = df[["src", "mt", "ref", "score"]]
            df["ref"] = df["ref"].astype(str)

        if self.word_level_training:
            df["mt_tags"] = convert_word_tags(df["mt_tags"].to_list())

        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype("float16")
        return df.to_dict("records")

    def prepare_target(
        self, mt_tags: List[List[int]], subword_masks: List[List[float]], max_len: int
    ):
        expanded_mt_tags = torch.sub(
            torch.zeros(subword_masks.size(0), max_len),
            torch.ones(subword_masks.size(0), max_len),
            alpha=1,
        )
        for k in range(len(subword_masks)):
            cnt = 0
            for j in range(len(subword_masks[k])):
                if subword_masks[k][j] > 0:
                    expanded_mt_tags[k][j] = mt_tags[k][cnt]
                    cnt += 1
        return expanded_mt_tags

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        :param stage: either 'fit', 'validate', 'test', or 'predict'
        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        if self.word_level_training:
            inputs = [self.encoder.prepare_sample_mt(sample["mt"])]
        else:
            inputs = [self.encoder.prepare_sample(sample["mt"])]

        if "src" in self.input_segments and "src" in sample:
            inputs.append(self.encoder.prepare_sample(sample["src"]))

        if "ref" in self.input_segments and "ref" in sample:
            inputs.append(self.encoder.prepare_sample(sample["ref"]))

        if self.word_level_training:
            contiguous_input, lengths, max_len = self.encoder.concat_sequences(
                inputs, word_outputs=True
            )
        else:
            contiguous_input = self.encoder.concat_sequences(inputs)

        if self.unite_training:
            qe_inputs = self.encoder.concat_sequences(
                [
                    self.encoder.prepare_sample_mt(sample["mt"]),
                    self.encoder.prepare_sample(sample["src"]),
                ]
            )
            metric_inputs = self.encoder.concat_sequences(
                [
                    self.encoder.prepare_sample_mt(sample["mt"]),
                    self.encoder.prepare_sample(sample["ref"]),
                ]
            )
            inputs = (qe_inputs, metric_inputs, contiguous_input)
        else:
            inputs = contiguous_input

        if stage == "predict":
            return inputs

        targets = Target(score=torch.tensor(sample["score"], dtype=torch.float))
        if self.loss.target_dim == 2 and stage == "fit":
            targets["std"] = torch.tensor(sample["std"], dtype=torch.float)

        # Read also word tags for the word tag prediction task
        if self.word_level_training:
            targets["tags"] = self.prepare_target(
                sample["mt_tags"], inputs["subwords_mask"], max_len
            )
        return inputs, targets

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        sentemb, wordemb = self.get_sentence_embedding(
            input_ids, attention_mask, return_word_embds=True
        )
        if self.word_level_training:
            sentence_output = self.estimator(sentemb)
            if self.hparams.use_rnn:
                rnn_out, _ = self.word_scorer_rnn(wordemb)
                word_output = self.hidden2tag(rnn_out)
            else:
                word_output = self.hidden2tag(wordemb)

            return Prediction(score=sentence_output.view(-1), tags=word_output)

        if self.loss.target_dim == 2:
            output = self.estimator(sentemb)
            return Prediction(score=output[:, 0], std=output[:, 1])

        return Prediction(score=self.estimator(sentemb).view(-1))

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

        def _forward(model, batch):
            if model.unite_training:
                outputs = [self(**i) for i in batch]
                if self.input_segments == ["mt", "src", "ref"] and len(batch) == 3:
                    return Prediction(
                        score=torch.mean(
                            torch.stack([o.score for o in outputs]), dim=0
                        ),
                        qe_score=outputs[0].score,
                        metric_score=outputs[1].score,
                        unified_score=outputs[2].score,
                    )
                else:
                    return Prediction(
                        score=torch.mean(torch.stack([o.score for o in outputs]), dim=0)
                    )
            else:
                model_outputs = self(**batch)
                # Recover word tags from subword 
                predicted_tags = model_outputs.tags.argmax(dim=2)
                subword_mask = batch['subwords_mask'] == 1
                word_tags = []
                for i in range(predicted_tags.shape[0]):
                    mask, tags = subword_mask[i, : ], predicted_tags[i, : ]
                    # TODO: Hard coded tag convertion
                    word_tags.append([
                        "OK" if y_hat != 1 else "BAD" 
                        for y_hat in torch.masked_select(tags, mask).tolist()
                    ])
                model_outputs.tags = word_tags
                return model_outputs

        if self.mc_dropout:
            raise NotImplementedError

        return _forward(self, batch)

    def initialize_loss(self) -> None:
        if self.hparams.loss == "kl":
            self.loss = KLLoss()
        elif self.hparams.loss == "mse":
            self.loss = MSELoss()
        else:
            raise Exception(
                "{} is not a valid loss function.".format(self.hparams.loss)
            )
        if self.hparams.word_level_training:
            self.wordloss = nn.CrossEntropyLoss(
                reduction="mean",
                weight=torch.tensor(self.hparams.word_weights),
                ignore_index=-1,
            )

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        if self.word_level_training:
            sentence_loss = self.loss(prediction.score, target.score)
            predictions = prediction.tags.view(-1, 2)
            targets = target.tags.view(-1).type(torch.LongTensor).cuda()
            word_loss = self.wordloss(predictions, targets)
            return sentence_loss * (1 - self.hparams.loss_lambda) + word_loss * (
                self.hparams.loss_lambda
            )

        if self.hparams.loss == "kl":
            return self.loss(prediction.score, prediction.std, target.score, target.std)

        return self.loss(prediction.score, target.score)

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
        # 1) We have a word layer mix: This means we will average word_embeddings across layers for
        #    word level classification.
        if self.word_layerwise_attention:
            if not self.training:
                n_splits = len(torch.split(encoder_out["all_layers"][-1], 8))
                word_embeddings = []
                for split in range(n_splits):
                    all_layers = []
                    for layer in range(len(encoder_out["all_layers"])):
                        layer_embs = torch.split(encoder_out["all_layers"][layer], 8)
                        all_layers.append(layer_embs[split])
                    split_attn = torch.split(attention_mask, 8)[split]
                    word_embeddings.append(self.layerwise_attention(all_layers, split_attn))
                word_embeddings = torch.cat(word_embeddings, dim=0)
            else:
                word_embeddings = self.layerwise_attention(
                    encoder_out["all_layers"], attention_mask
                )
        # 2) We DO NOT have a word layer mix and we want to use the embeddings from a specific layer!
        elif isinstance(self.hparams.word_layer, int) and 0 <= self.hparams.word_layer < self.encoder.num_layers:
            word_embeddings = encoder_out["all_layers"][self.hparams.word_layer]
    
        # 3) case where we want to used the same word_embeddings from sentence level.
        elif self.hparams.word_layer == "sent":
            word_embeddings = None

        # 4) The above cases failed!
        else:
            raise Exception("Invalid model word layer {}.".format(self.hparams.word_layer))

        # 1) We have a layer mix: This means we will average word embeddings across layers.
        #    These word embeddings will later be used to create a sentence-level embedding
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

        # 2) We DO NOT have a layer mix and we want to use the embeddings from a specific layer!
        #    These word embeddings will later be used to create a sentence-level embedding
        elif isinstance(self.hparams.sent_layer, int) and 0 <= self.hparams.sent_layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.hparams.sent_layer]

        # 3) The above cases failed!
        else:
            raise Exception("Invalid model sent layer {}.".format(self.hparams.sent_layer))

        if self.hparams.word_layer == "sent":
            word_embeddings = embeddings

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
            return sentemb, word_embeddings
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
        if self.unite_training:
            # In UniTE training is made of 3 losses:
            #    Lsrc + Lref + Lsrc+ref
            # For that reason we have to perform 3 forward passes and sum
            # the respective losses.
            qe_inputs, metric_inputs, unified_inputs = batch_input
            qe_prediction = self.forward(**qe_inputs)
            metric_prediction = self.forward(**metric_inputs)
            unified_prediction = self.forward(**unified_inputs)

            qe_loss = self.compute_loss(qe_prediction, batch_target)
            metric_loss = self.compute_loss(metric_prediction, batch_target)
            unified_loss = self.compute_loss(unified_prediction, batch_target)
            loss_value = qe_loss + metric_loss + unified_loss

        else:
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
        if self.unite_training:
            qe_inputs, metric_inputs, unified_inputs = batch_input
            qe_prediction = self.forward(**qe_inputs)
            metric_prediction = self.forward(**metric_inputs)
            unified_prediction = self.forward(**unified_inputs)
            scores = (
                qe_prediction.score + metric_prediction.score + unified_prediction.score
            ) / 3
        else:
            batch_prediction = self.forward(**batch_input)
            scores = batch_prediction.score

        if dataloader_idx == 0:
            self.train_metrics.update(
                scores, batch_target.score, batch_prediction.tags, batch_target.tags
            )

        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                scores, batch_target.score, batch_prediction.tags, batch_target.tags
            )
