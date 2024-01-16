#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# pip install transformers_stream_generator einops tiktoken
# export PYTHONPATH=$PWD/../../Qwen-7B-Chat:$PYTHONPATH
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

# QWEN_PATH = "../../Qwen-14B-Chat"
QWEN_PATH = "../../Qwen-7B-Chat"
folder = "./tmp/onnx"
device = torch.device("cuda:0")
origin_model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto").eval()
# origin_model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
transformer = origin_model.transformer
layers = transformer.h
NUM_LAYERS = len(layers)
SEQ_LENGTH = transformer.seq_length
HIDDEN_SIZE = layers[0].attn.hidden_size
NUM_HEADS = layers[0].attn.num_heads
for param in origin_model.parameters():
    param.requires_grad = False


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
        return out.float()


class QwenBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, 128)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, 128)

    def forward(self, hidden_states, position_ids, attention_mask):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            # registered_causal_mask=attention_mask,
            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states.float(), past_k.float(), past_v.float()


class QwenBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, 128)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, 128)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(past_k, past_v),
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            use_cache=True)
        k, v = past_kv
        return hidden_states.float(), k.float(), v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_qwen_block(layer_id):
    # input
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16().to(device)
    model = QwenBlock(layer_id)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/qwen_block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_qwen_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).bfloat16().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_HEADS, 128)).bfloat16().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_HEADS, 128)).bfloat16().to(device)
    model = QwenBlockCache(layer_id)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/qwen_block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input = torch.tensor([range(SEQ_LENGTH)]).to(device)
    torch.onnx.export(model, (input),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16().to(device)
    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)


def build_prompt(query):
    return f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'


def test_net_with_mask():
    embed = Embedding()
    blocks = [QwenBlock(i) for i in range(NUM_LAYERS)]
    block_kvs = [QwenBlockCache(i) for i in range(NUM_LAYERS)]
    query = '你好'
    print(query)
    promt = build_prompt(query)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out.bfloat16(), position_ids,
                              attention_mask.bfloat16())
        k[:, SEQ_LENGTH - token_len:] = k[:, :token_len]
        v[:, SEQ_LENGTH - token_len:] = v[:, :token_len]
        k[:, :SEQ_LENGTH - token_len] = 0
        v[:, :SEQ_LENGTH - token_len] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    token = lm(out.bfloat16()).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    while int(token) != tokenizer.im_end_id and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, :SEQ_LENGTH + 1 - token_len] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.bfloat16(), position_ids,
                                     attention_mask.bfloat16(),
                                     k_cache[i].bfloat16(), v_cache[i].bfloat16())
            k_tmp = torch.cat([k_cache[i][:, 1:], k], 1)
            v_tmp = torch.cat([v_cache[i][:, 1:], v], 1)
            k_cache[i][:] = k_tmp[:]
            v_cache[i][:] = v_tmp[:]
        token = lm(out.bfloat16()).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
for i in range(NUM_LAYERS):
    print("convert_block_{}".format(i))
    convert_qwen_block_cache(i)
    convert_qwen_block(i)

print("convert_embedding")
convert_embedding()

print("convert_lm_head")
convert_lm_head()
