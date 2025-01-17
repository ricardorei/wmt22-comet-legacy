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
import random
from typing import Callable, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
from transformers.utils import ModelOutput


class OrderedSampler(Sampler[int]):
    """
    Sampler that returns the indices in a deterministic order.
    """

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DataSampler(Dataset):

    def __init__(self, data_paths: List[str], read_func: Callable, gamma: float = 0.7):
        self.data = []
        for path in data_paths:
            self.data.append(read_func(path))

        self.sizes = [len(d) for d in self.data]
        self.total_size = sum(self.sizes)
        self.dist = np.array([s/self.total_size for s in self.sizes])**gamma
        self.dist = self.dist/sum(self.dist)

    def __getitem__(self, index):
        draw = np.random.choice(len(self.data), 1, p=self.dist)[0]
        if index >= len(self.data[draw]):
            index = index%len(self.data[draw])
        sample = self.data[draw][index]
        return sample
        
    def __len__(self):
        return max(self.sizes)


class Prediction(ModelOutput):
    "Renamed ModelOutput"
    pass

class Target(ModelOutput):
    "Renamed ModelOutput into Targets to keep same behaviour"
    pass
