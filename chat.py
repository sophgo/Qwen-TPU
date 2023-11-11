#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import datetime
import math
import unittest
import torch
import random
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_PATH = "/workspace/Qwen-7B-Chat"


def get_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, trust_remote_code=True).float().eval()
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH,
                                              trust_remote_code=True)
    return model, tokenizer


def main():
    model, tokenizer = get_model_and_tokenizer()
    print("Question:")
    history = []
    s = sys.stdin.readline().strip()
    while s != 'exit':
        print("Answer:")
        response, history = model.chat(tokenizer, s, history=history)
        print(response)
        print("Question:")
        s = sys.stdin.readline().strip()


if __name__ == '__main__':
    main()
