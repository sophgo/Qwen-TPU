#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

#pip install transformers_stream_generator einops tiktoken
#export PYTHONPATH=/workspace/Qwen-7B-Chat:$PYTHONPATH
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

QWEN_PATH = "/workspace/Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)


def build_prompt(query):
    return f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'


def test_encode():
    prompt = build_prompt("你好")
    ids = tokenizer.encode(prompt)
    print("input ids:{}".format(ids))


def test_decode():
    ids = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
        151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198
    ]
    s = tokenizer.decode(ids)
    print("str:{}".format(s))


test_encode()
