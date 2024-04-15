# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed embedding API."""

import distributed_embeddings.dist_model_parallel
from distributed_embeddings.dist_model_parallel import DistEmbeddingStrategy
from distributed_embeddings.dist_model_parallel import DistributedEmbedding
from distributed_embeddings.dist_model_parallel import broadcast_variables
from distributed_embeddings.dist_model_parallel import DistributedGradientTape
from distributed_embeddings.dist_model_parallel import DistributedOptimizer
from distributed_embeddings.dist_model_parallel import BroadcastGlobalVariablesCallback
