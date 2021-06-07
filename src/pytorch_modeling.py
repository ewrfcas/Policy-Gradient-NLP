# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""
from __future__ import print_function

import copy
import json
import math
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

BertLayerNorm = nn.LayerNorm

SPIECE_UNDERLINE = '▁'

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def fast_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"fast_gelu": fast_gelu, "gelu": gelu, "relu": torch.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln_type = 'postln'
        if 'ln_type' in config.__dict__:
            self.ln_type = config.ln_type

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.ln_type == 'preln':
            hidden_states = hidden_states + input_tensor
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.ln_type = 'postln'
        if 'ln_type' in config.__dict__:
            self.ln_type = config.ln_type

    def forward(self, input_tensor, attention_mask):
        if self.ln_type == 'preln':
            hidden_state = self.output.LayerNorm(input_tensor)  # pre_ln
            self_output = self.self(hidden_state, attention_mask)
        else:
            self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln_type = 'postln'
        if 'ln_type' in config.__dict__:
            self.ln_type = config.ln_type

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.ln_type == 'preln':
            hidden_states = hidden_states + input_tensor
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.ln_type = 'postln'
        if 'ln_type' in config.__dict__:
            self.ln_type = config.ln_type
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        if self.ln_type == 'preln':
            attention_output_pre = self.output.LayerNorm(attention_output)
        else:
            attention_output_pre = attention_output
        intermediate_output = self.intermediate(attention_output_pre)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class UniLM(PreTrainedBertModel):
    def __init__(self, config):
        super(UniLM, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_att_mask(self, input_mask, context_mask):
        # [bs, len]->[bs, 1, 1, len]
        input_mask = input_mask.unsqueeze(1).unsqueeze(2)
        input_mask = input_mask.repeat(1, 1, input_mask.shape[-1], 1)
        context_mask = context_mask.unsqueeze(1).unsqueeze(2)
        context_mask = context_mask.repeat(1, 1, context_mask.shape[-1], 1)

        attention_mask = torch.tril(input_mask) + context_mask
        attention_mask = attention_mask.clamp(0, 1)

        return attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, context_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = self.get_att_mask(attention_mask, context_mask)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMaskedLM(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForMultiCLS(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMultiCLS, self).__init__(config)
        self.bert = BertModel(config)
        self.cls_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pos=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        pos = pos.unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        sequence_output = torch.gather(sequence_output, dim=1, index=pos)
        logits = self.cls_outputs(sequence_output)

        if labels is not None:
            # classifier loss
            loss_fct_cls = CrossEntropyLoss(ignore_index=-1)  # no loss for has answer
            loss = loss_fct_cls(logits.reshape(-1, 2), labels.reshape(-1))
            return loss
        else:
            return logits


class BertForClassification(PreTrainedBertModel):
    def __init__(self, config, class_num=2):
        super(BertForClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, class_num)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        logits = self.cls(pooled_output)  # [bs, cls_num]

        if labels is not None:
            # classifier loss
            loss_fct_cls = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct_cls(logits, labels)
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


import numpy as np


class BertForQA_PG(PreTrainedBertModel):
    def __init__(self, config, rl_weight=0.3, trainable_weight=False):
        super(BertForQA_PG, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        # self.apply(self.init_bert_weights)
        self.rl_weight = rl_weight
        self.trainable_weight = trainable_weight
        if self.trainable_weight:
            self.sigma_ce = nn.Parameter(torch.tensor(1 / np.sqrt(2), dtype=torch.float32), requires_grad=True)
            self.sigma_rl = nn.Parameter(torch.tensor(1 / np.sqrt(2), dtype=torch.float32), requires_grad=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, pg_weight=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # get start-end seqs s:[0,1,0,0,0] e:[0,0,0,0,1] gt:[0,1,1,1,1]
            batch, length = input_ids.shape[0], input_ids.shape[1]
            start_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, start_positions[:, None], 1)
            end_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, end_positions[:, None], 1)
            gt_cumsum_start = torch.cumsum(start_onehot, dim=1)
            gt_cumsum_end = torch.cumsum(end_onehot, dim=1)
            gt_cumsum = torch.clamp(gt_cumsum_start - gt_cumsum_end + end_onehot, 0, 1.0)

            # get greed cumsums
            start_logits_ = start_logits.clone()
            start_logits_[:, 0] = -20000.0
            start_logits_ = start_logits_ - (1 - attention_mask) * 20000.0
            end_logits_ = end_logits.clone()
            end_logits_[:, 0] = -20000.0
            end_logits_ = end_logits_ - (1 - attention_mask) * 20000.0

            greed_start = torch.argmax(start_logits_, dim=1)
            greed_start_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, greed_start[:, None], 1)
            greed_cumsum_start = torch.cumsum(greed_start_onehot, dim=1)
            # mask all pos before start
            end_logits_2 = end_logits_ - (1 - greed_cumsum_start) * 20000.0  # [b,len]
            greed_end = torch.argmax(end_logits_2, dim=1)
            greed_end_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, greed_end[:, None], 1)
            greed_cumsum_end = torch.cumsum(greed_end_onehot, dim=1)
            greed_cumsum = torch.clamp(greed_cumsum_start - greed_cumsum_end + greed_end_onehot, 0, 1.0)
            greed_tp = torch.sum(torch.clamp(greed_cumsum + gt_cumsum - 1.0, 0, 1.0), dim=1)
            greed_precision = greed_tp / (torch.sum(greed_cumsum, dim=1) + 1e-6)
            greed_recall = greed_tp / (torch.sum(gt_cumsum, dim=1) + 1e-6)
            greed_f1 = (2 * greed_precision * greed_recall) / (greed_precision + greed_recall + 1e-6)

            # get sample cumsums
            sample_start = torch.multinomial(torch.softmax(start_logits_, dim=1), 1)
            sample_start_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, sample_start, 1)
            sample_cumsum_start = torch.cumsum(sample_start_onehot, dim=1)
            # mask all pos before start
            end_logits_2 = end_logits_ - (1 - greed_cumsum_start) * 20000.0  # [b,len]
            sample_end = torch.multinomial(torch.softmax(end_logits_2, dim=1), 1)
            sample_end_onehot = torch.zeros((batch, length), dtype=logits.dtype, device=logits.device). \
                scatter_(1, sample_end, 1)
            sample_cumsum_end = torch.cumsum(sample_end_onehot, dim=1)
            sample_cumsum = torch.clamp(sample_cumsum_start - sample_cumsum_end + sample_end_onehot, 0, 1.0)
            sample_tp = torch.sum(torch.clamp(sample_cumsum + gt_cumsum - 1.0, 0, 1.0), dim=1)
            sample_precision = sample_tp / (torch.sum(sample_cumsum, dim=1) + 1e-6)
            sample_recall = greed_tp / (torch.sum(gt_cumsum, dim=1) + 1e-6)
            sample_f1 = (2 * sample_precision * sample_recall) / (sample_precision + sample_recall + 1e-6)

            # 规范化rewards
            ML_reward = torch.clamp_min(sample_f1 - greed_f1, 0.).detach()
            GD_reward = torch.clamp_min(greed_f1 - sample_f1, 0.).detach()

            rl_loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            ML_start_loss = rl_loss_fct(start_logits, sample_start.squeeze(-1)) * ML_reward
            ML_end_loss = rl_loss_fct(end_logits, sample_end.squeeze(-1)) * ML_reward
            ML_loss = (ML_start_loss.mean() + ML_end_loss.mean()) / 2
            GD_start_loss = rl_loss_fct(start_logits, greed_start.squeeze(-1)) * GD_reward
            GD_end_loss = rl_loss_fct(end_logits, greed_end.squeeze(-1)) * GD_reward
            GD_loss = (GD_start_loss.mean() + GD_end_loss.mean()) / 2
            RL_loss = ML_loss + GD_loss

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if self.trainable_weight:
                total_loss = (1 / (2 * (self.sigma_ce ** 2) + 1e-7)) * total_loss + \
                             (1 / (2 * (self.sigma_rl ** 2) + 1e-7)) * RL_loss + \
                             torch.log(self.sigma_ce ** 2 + 1e-7) + torch.log(self.sigma_rl ** 2 + 1e-7)
                return total_loss, (1 / (2 * (self.sigma_rl ** 2) + 1e-7)) * RL_loss
            elif pg_weight is None:
                total_loss = total_loss + self.rl_weight * RL_loss
                return total_loss, self.rl_weight * RL_loss
            else:
                total_loss = total_loss + pg_weight * RL_loss
                return total_loss, pg_weight * RL_loss
        else:
            return start_logits, end_logits


class BertForUniLMInference(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForUniLMInference, self).__init__(config)
        self.bert = UniLM(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                context_mask=None, output_idx=None, vocab_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       context_mask=context_mask,
                                       output_all_encoded_layers=False)
        output_idx = output_idx.unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        sequence_output = torch.gather(sequence_output, dim=1, index=output_idx)
        logits = self.cls(sequence_output)
        if vocab_mask is not None:
            # [bs, vocab_size]->[bs, len2, vocab_size]
            vocab_mask = vocab_mask.reshape(vocab_mask.shape[0], 1, vocab_mask.shape[1]).repeat(1, logits.shape[1], 1)
            vocab_mask = (1. - vocab_mask.to(dtype=logits.dtype)) * -10000.0
            logits = logits + vocab_mask
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return logits


class BertForUniLM(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForUniLM, self).__init__(config)
        self.bert = UniLM(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                context_mask=None, output_idx=None, output_ids=None, vocab_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       context_mask=context_mask,
                                       output_all_encoded_layers=False)
        output_idx = output_idx.unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        sequence_output = torch.gather(sequence_output, dim=1, index=output_idx)
        logits = self.cls(sequence_output)  # [bs, len2, vocab_size]
        if vocab_mask is not None:
            # [bs, vocab_size]->[bs, len2, vocab_size]
            vocab_mask = vocab_mask.reshape(vocab_mask.shape[0], 1, vocab_mask.shape[1]).repeat(1, logits.shape[1], 1)
            vocab_mask = (1. - vocab_mask.to(dtype=logits.dtype)) * -10000.0
            logits = logits + vocab_mask
        if output_ids is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.reshape(-1, logits.shape[-1]), output_ids.reshape(-1))
            return loss
        else:
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            return logits


class BertForPosCLS(PreTrainedBertModel):
    def __init__(self, config, class_num=260):
        super(BertForPosCLS, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = torch.nn.Linear(config.hidden_size, class_num)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        logits = self.cls(sequence_output)  # [bs, len, len]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            return loss
        else:
            return logits


class BertForUniLM_PolicyGradient(PreTrainedBertModel):
    def __init__(self, config, rl_weight=0.3, tokenizer=None):
        super(BertForUniLM_PolicyGradient, self).__init__(config)
        self.bert = UniLM(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.rl_weight = rl_weight
        self.tokenizer = tokenizer

    def get_ter_score(self, y_true, y_pred):
        dp = [[0 for _ in range(len(y_pred) + 1)] for _ in range(len(y_true) + 1)]
        for i in range(1, len(dp)):
            dp[i][0] = i
        for i in range(1, len(dp[0])):
            dp[0][i] = i
        for i in range(1, len(dp)):
            for j in range(1, len(dp[i])):
                if y_true[i - 1] == y_pred[j - 1]:
                    d = 0
                else:
                    d = 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + d)
        ter = dp[-1][-1] / (len(y_true) + 1e-5)
        return 1.0 - ter

    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(self, c):
        if c in ('。', '，', ',', '！', '!', '？', '?', '；', ';', '、', '：', ':', '（', '(', '）', ')', '－', '~', '～',
                 '「', '」', '《', '》', '"', '“', '”', '$', '『', '』', '—', '-', '‘', '’', '\'', '[', '【',
                 ']', '】', '=', '|', '<', '>'):
            return True
        return False

    def combine_tokens(self, tokens):
        output_text = []
        for tok in tokens:
            if len(tok) == 1 and self._is_chinese_char(ord(tok)):
                output_text.append(tok)
            elif tok.startswith('##'):
                output_text.append(tok.replace('##', ''))
            elif len(output_text) > 0:
                if len(output_text[-1]) == 1 and (
                        self._is_chinese_char(ord(output_text[-1])) or self.is_fuhao(output_text[-1])):
                    output_text.append(tok)
                else:
                    output_text.append(' ' + tok)
            else:
                output_text.append(tok)

        output_text = "".join(output_text).strip()
        return output_text

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp) or self.is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    def split_tokens(self, text):
        context_chs = self._tokenize_chinese_chars(text)
        doc_tokens = []
        prev_is_whitespace = True
        for c in context_chs:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
        return doc_tokens

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                context_mask=None, output_idx=None, output_ids=None, texts=None):
        # texts: 真实的文本，计算TER-SCORE
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       context_mask=context_mask,
                                       output_all_encoded_layers=False)
        output_idx = output_idx.unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        sequence_output = torch.gather(sequence_output, dim=1, index=output_idx)
        logits = self.cls(sequence_output)  # [bs, len2, vocab_size]
        if output_ids is not None and texts is not None:
            # 训练阶段
            probs = torch.softmax(logits, dim=2)  # [bs, len2, vocab_size]
            (bs, length, vocab_size) = probs.shape
            # 先求greed search的结果
            greed_preds = torch.argmax(probs, dim=2)  # [bs, len2]
            greed_preds_cpu = greed_preds.cpu().detach().numpy()
            greed_texts = []
            validate_length = torch.sum(torch.clamp(output_ids, 0, 1), dim=1) - 1  # [bs,] - [END]
            validate_length = validate_length.cpu().detach().numpy()
            for i in range(greed_preds_cpu.shape[0]):
                greed_texts.append(
                    self.combine_tokens(self.tokenizer.convert_ids_to_tokens(greed_preds_cpu[i, :validate_length[i]])))

            # 求multinomial search结果
            probs_reshape = probs.reshape(-1, vocab_size)  # [bs*len2, vocab_size]
            multinomial_preds = torch.multinomial(probs_reshape, 1)  # [bs*len2]
            multinomial_preds_cpu = multinomial_preds.reshape(bs, length).cpu().detach().numpy()  # [bs, len2]
            multinomial_texts = []
            for i in range(multinomial_preds_cpu.shape[0]):
                multinomial_texts.append(
                    self.combine_tokens(
                        self.tokenizer.convert_ids_to_tokens(multinomial_preds_cpu[i, :validate_length[i]])))

            # 计算TER分数作为rewards
            greed_rewards = []
            multinomial_rewards = []
            for greed_text, multinomial_text, text in zip(greed_texts, multinomial_texts, texts):
                text_tokens = self.split_tokens(text)
                greed_rewards.append(self.get_ter_score(self.split_tokens(greed_text), text_tokens))
                multinomial_rewards.append(self.get_ter_score(self.split_tokens(multinomial_text), text_tokens))

            # 规范化rewards
            greed_rewards = torch.tensor(greed_rewards).to(dtype=logits.dtype, device=logits.device)
            multinomial_rewards = torch.tensor(multinomial_rewards).to(dtype=logits.dtype, device=logits.device)
            # [bs, len2]
            ML_reward = torch.clamp(multinomial_rewards - greed_rewards, 0., 1e4).reshape(-1, 1).repeat(1, length)
            GD_reward = torch.clamp(greed_rewards - multinomial_rewards, 0., 1e4).reshape(-1, 1).repeat(1, length)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            CE_loss = loss_fct(logits.reshape(-1, vocab_size), output_ids.reshape(-1))

            output_mask = torch.clamp(output_ids, 0, 1).to(dtype=logits.dtype)
            rl_loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            ML_loss = rl_loss_fct(logits.reshape(-1, vocab_size), multinomial_preds.reshape(-1))  # [bs*len2]
            GD_loss = rl_loss_fct(logits.reshape(-1, vocab_size), greed_preds.reshape(-1))
            ML_loss = ML_loss.reshape(bs, length)
            GD_loss = GD_loss.reshape(bs, length)
            ML_loss = torch.sum(ML_loss * ML_reward * output_mask) / torch.sum(output_mask)
            GD_loss = torch.sum(GD_loss * GD_reward * output_mask) / torch.sum(output_mask)
            RL_loss = ML_loss + GD_loss

            loss = CE_loss + self.rl_weight * RL_loss

            return loss, CE_loss, self.rl_weight * RL_loss
        else:
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            return logits
