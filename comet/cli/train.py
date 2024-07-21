#!/usr/bin/env python3

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
"""

Command for training new Metrics.
=================================

e.g:
```
    comet-train --cfg configs/models/regression_metric.yaml
```

For more details run the following command:
```
    comet-train --help
```
"""
import json
import logging
import warnings
import math

import optuna
from comet.models import (CSPECMetric, RankingMetric, ReferencelessRegression,
                          RegressionMetric, UniTEMetric, UniTEMetricMT)
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(
        ReferencelessRegression, "referenceless_regression_metric"
    )
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(UniTEMetric, "unite_metric")
    parser.add_subclass_arguments(UniTEMetricMT, "unite_metric_multi_task")
    parser.add_subclass_arguments(CSPECMetric, "cspec_metric")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load_from_checkpoint",
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict_load", 
        action="store_true", 
        help="Strictly enforce that the keys in checkpoint_path match the keys returned by this module's state dict."
    )
    parser.add_argument(
        "--optuna_search",
        action="store_true",
        help="If used we will launch an Optuna Search for some basic hyper parameters such as learning rate."
    )
    return parser

def initialize_trainer(configs) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(configs.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(configs.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor]
    print("TRAINER ARGUMENTS: ")
    print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)
    return trainer

def initialize_model(configs):
    print("MODEL ARGUMENTS: ")
    if configs.regression_metric is not None:
        print(
            json.dumps(
                configs.regression_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = RegressionMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.regression_metric.init_args),
            )
        else:
            model = RegressionMetric(**namespace_to_dict(configs.regression_metric.init_args))
    elif configs.referenceless_regression_metric is not None:
        print(
            json.dumps(
                configs.referenceless_regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.referenceless_regression_metric.init_args),
            )
        else:
            model = ReferencelessRegression(
                **namespace_to_dict(configs.referenceless_regression_metric.init_args)
            )
    elif configs.ranking_metric is not None:
        print(
            json.dumps(
                configs.ranking_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = ReferencelessRegression.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.ranking_metric.init_args),
            )
        else:
            model = RankingMetric(**namespace_to_dict(configs.ranking_metric.init_args))
    elif configs.unite_metric is not None:
        print(
            json.dumps(
                configs.unite_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = UniTEMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.unite_metric.init_args),
            )
        else:
            model = UniTEMetric(**namespace_to_dict(configs.unite_metric.init_args))
        #model = UniTEMetric(**namespace_to_dict(cfg.unite_metric.init_args))
    elif configs.unite_metric_multi_task is not None:
        print(
            json.dumps(
                configs.unite_metric_multi_task.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        #model = UniTEMetricMT(**namespace_to_dict(cfg.unite_metric_multi_task.init_args))
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = UniTEMetricMT.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.unite_metric_multi_task.init_args),
            )
        else:
            model = UniTEMetricMT(**namespace_to_dict(configs.unite_metric_multi_task.init_args))
    elif configs.cspec_metric is not None:
        print(
            json.dumps(
                configs.cspec_metric.init_args, indent=4, default=lambda x: x.__dict__
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = CSPECMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.cspec_metric.init_args),
            )
        else:
            model = CSPECMetric(**namespace_to_dict(configs.cspec_metric.init_args))
    else:
        raise Exception("Model configurations missing!")

    return model

def objective(trial: optuna.trial.Trial, configs: Namespace) -> float:
    model_configs = None
    if configs.regression_metric is not None:
        model_configs = configs.regression_metric.init_args
    elif configs.referenceless_regression_metric is not None:
        model_configs = configs.referenceless_regression_metric.init_args
    elif configs.ranking_metric is not None:
        model_configs = configs.ranking_metric.init_args
    elif configs.unite_metric is not None:
        model_configs = configs.unite_metric.init_args
    elif configs.unite_metric_multi_task is not None:
        model_configs = configs.unite_metric_multi_task.init_args
    elif configs.cspec_metric is not None:
        model_configs = configs.cspec_metric.init_args
    else:
        raise Exception("Model configurations missing!")

    # ADD WHICH PARAMETERS YOU WANT TO SEARCH HERE!

    #model_configs.batch_size = trial.suggest_int("batch_size", 1, 8)
    model_configs.loss_lambda = trial.suggest_uniform("loss_lambda", 0.3, 0.7)
    bad_weight = trial.suggest_uniform("bad_weight", 0.5, 0.95)
    model_configs.word_weights[0] = 1-bad_weight
    model_configs.word_weights[1] = bad_weight
    model_configs.nr_frozen_epochs = trial.suggest_uniform("nr_frozen_epochs", .0, 1.0)
    model_configs.learning_rate = trial.suggest_uniform("learning_rate", 1e-06, 8e-05)
    model_configs.encoder_learning_rate = trial.suggest_uniform("encoder_learning_rate", 1e-07, 1e-05)
    model = initialize_model(configs)

    trainer = initialize_trainer(configs)
    trainer.logger.log_hyperparams(model_configs)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )
    trainer.fit(model)
    best_score = trainer.callbacks[0].best_score.item()
    return -1 if math.isnan(best_score) else best_score

def train_command() -> None:
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    if cfg.optuna_search:
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        try:
            study.optimize(lambda t: objective(t, cfg), n_trials=30)

        except KeyboardInterrupt:
            print("Early stopping search caused by ctrl-C")

        except Exception as e:
            print(f"Error occured during search: {e}; current best params are {study.best_params}")

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        trainer = initialize_trainer(cfg)
        model = initialize_model(cfg)
        # Related to train/val_dataloaders:
        # 2 workers per gpu is enough! If set to the number of cpus on this machine
        # it throws another exception saying its too many workers.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer.fit(model)


if __name__ == "__main__":
    train_command()
