# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Activations package definition."""
from models.modeling.activations.gelu import gelu
from models.modeling.activations.relu import relu6
from models.modeling.activations.sigmoid import hard_sigmoid
from models.modeling.activations.swish import hard_swish
from models.modeling.activations.swish import identity
from models.modeling.activations.swish import simple_swish
