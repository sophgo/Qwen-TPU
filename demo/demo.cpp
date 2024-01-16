//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "memory.h"
#include "tokenizer.h"
#include "bmruntime_interface.h"
#include <getopt.h>

static const uint16_t BF16_NEG_10000 = 0xC61C; // -9984 by bfloat16

static const std::string TOKENIZER_MODEL = "qwen.tiktoken";

class QwenChat {
public:
  void init(const std::vector<int> &devid, std::string model);
  void chat();
  void deinit();

private:
  void answer(const std::string &input_str);
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void move2end(const bm_tensor_t &kv);
  void load_tiktoken();

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm;
  bm_tensor_t inputs_embed1, outputs_embed1;
  bm_tensor_t inputs_lm, outputs_lm;
  bm_tensor_t inputs_pid, next_pid, inputs_attention, next_attention;
  std::vector<bm_tensor_t> past_key;
  std::vector<bm_tensor_t> past_value;
  bm_tensor_t present_key_cache, present_value_cache;
  std::string name_embed;
  std::string name_lm;
  std::vector<std::string> name_blocks;
  std::vector<std::string> name_blocks_cache;
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  std::unique_ptr<QwenTokenizer> tk;
  std::vector<std::string> history;
};

void QwenChat::load_tiktoken() {
  printf("Load %s ... \n", TOKENIZER_MODEL.c_str());
  tk = std::make_unique<QwenTokenizer>(TOKENIZER_MODEL);
}

void QwenChat::init(const std::vector<int> &devices, std::string model) {
  load_tiktoken();
  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];
// create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model.c_str());
  assert(true == ret);
  printf("Done!\n");
  // net embed and lm_head
  name_embed = "embedding";
  name_lm = "lm_head";
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  assert(net_embed->stage_num == 2);
  SEQLEN = net_embed->stages[1].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 2) / 2;
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "qwen_block_" + std::to_string(i);
    auto cache_name = "qwen_block_cache_" + std::to_string(i);
    name_blocks.emplace_back(block_name);
    name_blocks_cache.emplace_back(cache_name);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // net device mem
  ret = bmrt_tensor(&inputs_embed1, p_bmrt, net_embed->input_dtypes[0],
                    net_embed->stages[1].input_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&outputs_embed1, p_bmrt, net_embed->output_dtypes[0],
                    net_embed->stages[1].output_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_pid, p_bmrt, net_blocks[0]->input_dtypes[1],
                    net_blocks[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_attention, p_bmrt, net_blocks[0]->input_dtypes[2],
                    net_blocks[0]->stages[0].input_shapes[2]);
  assert(true == ret);

  ret = bmrt_tensor(&next_pid, p_bmrt, net_blocks_cache[0]->input_dtypes[1],
                    net_blocks_cache[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret =
      bmrt_tensor(&next_attention, p_bmrt, net_blocks_cache[0]->input_dtypes[2],
                  net_blocks_cache[0]->stages[0].input_shapes[2]);
  assert(true == ret);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    ret = bmrt_tensor(&past_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret = bmrt_tensor(&past_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                      net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
  }
  ret = bmrt_tensor(&present_key_cache, p_bmrt,
                    net_blocks_cache[0]->output_dtypes[1],
                    net_blocks_cache[0]->stages[0].output_shapes[1]);
  assert(true == ret);
  ret = bmrt_tensor(&present_value_cache, p_bmrt,
                    net_blocks_cache[0]->output_dtypes[2],
                    net_blocks_cache[0]->stages[0].output_shapes[2]);
  assert(true == ret);
  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}

void QwenChat::deinit() {
  bm_free_device(bm_handle, inputs_embed1.device_mem);
  bm_free_device(bm_handle, outputs_embed1.device_mem);
  bm_free_device(bm_handle, inputs_lm.device_mem);
  bm_free_device(bm_handle, outputs_lm.device_mem);
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  bm_free_device(bm_handle, present_key_cache.device_mem);
  bm_free_device(bm_handle, present_value_cache.device_mem);
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, past_key[i].device_mem);
    bm_free_device(bm_handle, past_value[i].device_mem);
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

// after first block, move real result to end of mem
void QwenChat::move2end(const bm_tensor_t &kv) {
  if (token_length >= SEQLEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / SEQLEN;
  auto real_size = token_length * bytes;
  auto mem =
      bm_mem_from_device(bm_mem_get_device_addr(kv.device_mem), real_size);
  auto buffer = new uint8_t[real_size];
  auto dst = new uint8_t[total_size];
  bm_memcpy_d2s(bm_handle, (void *)buffer, mem);
  memset(dst, 0, total_size - real_size);
  memcpy(dst + total_size - real_size, buffer, real_size);
  bm_memcpy_s2d(bm_handle, kv.device_mem, (void *)dst);
  delete[] buffer;
  delete[] dst;
}

int QwenChat::forward_first(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, BF16_NEG_10000);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  bm_memcpy_s2d(bm_handle, inputs_embed1.device_mem, (void *)input_ids.data());
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &inputs_embed1,
                                   1, &outputs_embed1, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem,
                (void *)attention_mask.data());
  auto inputs_embed = outputs_embed1;
  bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(), inputs_block, 3,
                                outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }
  int bytes = inputs_embed.device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm.device_mem, 0,
                     inputs_embed.device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

int QwenChat::forward_next() {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  // for (int i = 0; i <= SEQLEN - token_length; i++) {
  //   attention_mask[i] = BF16_NEG_10000;
  // }
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = BF16_NEG_10000;
  }
  int32_t position_id = token_length - 1;
  // embedding
  outputs_lm.shape = net_embed->stages[0].input_shapes[0];
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &outputs_lm, 1,
                                   &inputs_lm, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
  // blocks
  bm_memcpy_s2d(bm_handle, next_attention.device_mem,
                (void *)attention_mask.data());
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void *)&position_id);
  auto inputs_embed = inputs_lm;
  inputs_embed.shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  int bytes = bm_mem_get_device_size(present_key_cache.device_mem);
  int token_offset = (token_length - 1) * bytes;
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t inputs_block[5] = {inputs_embed, next_pid, next_attention,
                                   past_key[i], past_value[i]};
    bm_tensor_t outputs_block[3] = {inputs_embed, present_key_cache,
                                    present_value_cache};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block, 5, outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    bm_memcpy_d2d_byte(bm_handle, past_key[i].device_mem, token_offset,
                       present_key_cache.device_mem, 0, bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[i].device_mem, token_offset,
                       present_value_cache.device_mem, 0, bytes);
  }
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

void QwenChat::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str.empty()) {
      continue;
    }
    if (input_str == "exit" || input_str == "quit") {
      break;
    }
    if (input_str == "clear") {
      history.clear();
      continue;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void QwenChat::answer(const std::string &input_str) {
  int tok_num = 0;
  history.emplace_back(std::move(input_str));
  auto input_ids = tk->encode_history(history, SEQLEN);
  token_length = input_ids.size();
  auto time_1 = std::chrono::system_clock::now();
  int pre_token = 0;
  int token = forward_first(input_ids);
  auto time_2 = std::chrono::system_clock::now();
  std::string result;
  while (token != tk->im_end_id && token_length < SEQLEN) {
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    auto pre_word = tk->decode(pre_ids);
    auto word = tk->decode(ids);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    std::cout << diff << std::flush;
    if (token_length < SEQLEN) {
      token_length++;
    }
    tok_num++;
    token = forward_next();
  }
  auto time_3 = std::chrono::system_clock::now();
  auto ftl_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(time_2 - time_1);
  auto tps_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(time_3 - time_2);
  double tps = tok_num / (tps_dur.count() * 1e-6);
  if (token_length >= SEQLEN) {
    printf(" ......\nWarning: cleanup early history\n");
  }
  // double tht = tokens.size() / (tht_dur.count() * 1e-6);
  printf("\nFTL:%f s, TPS: %f tokens/s\n", ftl_dur.count() * 1e-6, tps);
  history.emplace_back(result);
  if (token_length + 128 >= SEQLEN) {
    int num = (history.size() + 3) / 4 * 2;
    history.erase(history.begin(), history.begin() + num);
  }
}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  printf("Usage:\n"
         "  --help         : Show help info.\n"
         "  --model        : Set model path \n"
         "  --devid        : Set devices to run for model, e.g. 1,2. if not "
         "set, use 0\n");
}

void processArguments(int argc, char *argv[], std::string &qwen_model,
                      std::vector<int> &devices) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:d:h:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      qwen_model = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_SUCCESS);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for QwenChat in BM1684X\n");
  std::string qwen_model;
  std::vector<int> devices = {0};
  processArguments(argc, argv, qwen_model, devices);
  if (qwen_model.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  QwenChat qwen;
  printf("Init Environment ...\n");
  qwen.init(devices, qwen_model);
  printf("==========================\n");
  qwen.chat();
  qwen.deinit();
  return 0;
}
