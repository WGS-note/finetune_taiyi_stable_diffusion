# coding=utf-8
# Copyright 2022 IDEA-CCNL and The HuggingFace Inc. team. All rights reserved.
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
""" TransfoXLDenoise model configuration """

from transformers.configuration_utils import PretrainedConfig


Transfo_XL_Denoise_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "transformer-xl-1b-base": "https://huggingface.co/transformer-xl-1b-base/resolve/main/config.json",
    # See all TransfoXLDenoise models at https://huggingface.co/models?filter=transfo_xl_denoise
}


class TransfoXLDenoiseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~TransfoXLDenoiseModel`].
    It is used to instantiate an TransfoXLDenoise model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the TransfoXLDenoise [transformer-xl-1b-base](https://huggingface.co/transformer-xl-1b-base) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the TransfoXLDenoise model. Defines the number of different
            tokens that can be represented by the
            `inputs_ids` passed when calling [`~TransfoXLDenoiseModel`] or
            [`~TFTransfoXLDenoiseModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~TransfoXLDenoiseModel`] or
            [`~TFTransfoXLDenoiseModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import TransfoXLDenoiseModel, TransfoXLDenoiseConfig

    >>> # Initializing a TransfoXLDenoise transformer-xl-1b-base style configuration
    >>> configuration = TransfoXLDenoiseConfig()

    >>> # Initializing a model from the transformer-xl-1b-base style configuration
    >>> model = TransfoXLDenoiseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "transfo_xl_denoise"

    def __init__(
        self,
        num_layers=32,
        vocab_size=50048,
        hidden_size=1600,
        num_attention_heads=25,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=512,
        max_memory_length=512,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=True,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        super().__init__(**kwargs)
