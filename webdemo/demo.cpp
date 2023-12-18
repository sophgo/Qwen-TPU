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

static const int NUM_LAYERS = 32;
static const int MAX_LEN = 512;
static const int HIDDEN_SIZE = 4096;
static const uint16_t BF16_NEG_10000 = 0xC61C; // -9984 by bfloat16

static const std::string TOKENIZER_MODEL = "qwen.tiktoken";

class QwenChat {
public:
  void init(const std::vector<int> &devid, std::string model, std::string tik_path="");
  void chat();
  void deinit();

public:
  void answer(const std::string &input_str);
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void move2end(const bm_tensor_t &kv);
  void load_tiktoken();
  void load_tiktoken(std::string tik_path);

public:
  std::string name_embed;
  std::string name_lm;
  std::string name_blocks[NUM_LAYERS];
  std::string name_blocks_cache[NUM_LAYERS];
  int token_length;
  std::unique_ptr<QwenTokenizer> tk;
  std::vector<std::string> history;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_blocks[NUM_LAYERS];
  const bm_net_info_t *net_blocks_cache[NUM_LAYERS];
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm;
  bm_tensor_t inputs_embed_512, outputs_embed_512;
  bm_tensor_t inputs_lm, outputs_lm;
  bm_tensor_t inputs_pid, next_pid, inputs_attention, next_attention;
  bm_tensor_t past_key[NUM_LAYERS], past_value[NUM_LAYERS];
};

void QwenChat::load_tiktoken() {
  printf("Load %s ... \n", TOKENIZER_MODEL.c_str());
  tk = std::make_unique<QwenTokenizer>(TOKENIZER_MODEL);
}

void QwenChat::load_tiktoken(std::string tik_path) {
  tk = std::make_unique<QwenTokenizer>(tik_path);
}

void QwenChat::init(const std::vector<int> &devices, std::string model, std::string tik_path) {
  if(tik_path==""){
    load_tiktoken();
  }else{
    load_tiktoken(tik_path);
  }
  // request bm_handle
    //   std::cout << "Device [ ";
    //   for (auto d : devices) {
    //     std::cout << d << " ";
    //   }
    //   std::cout << "] loading ....\n";
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
  // printf("Model[%s] loading ....\n", model.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model.c_str());
  assert(true == ret);
  // printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "qwen_block_" + std::to_string(i);
    name_blocks_cache[i] = "qwen_block_cache_" + std::to_string(i);
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks[i] = bmrt_get_network_info(p_bmrt, name_blocks[i].c_str());
    net_blocks_cache[i] =
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str());
  }

  // net device mem
  ret = bmrt_tensor(&inputs_embed_512, p_bmrt, net_embed->input_dtypes[0],
                    net_embed->stages[1].input_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&outputs_embed_512, p_bmrt, net_embed->output_dtypes[0],
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

  for (int i = 0; i < NUM_LAYERS; i++) {
    ret = bmrt_tensor(&past_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret = bmrt_tensor(&past_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                      net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
  }
  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}

void QwenChat::deinit() {
  bm_free_device(bm_handle, inputs_embed_512.device_mem);
  bm_free_device(bm_handle, outputs_embed_512.device_mem);
  bm_free_device(bm_handle, inputs_lm.device_mem);
  bm_free_device(bm_handle, outputs_lm.device_mem);
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
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
  if (token_length >= MAX_LEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / MAX_LEN;
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
  std::vector<int> input_ids(MAX_LEN, 0);
  std::vector<int> position_id(MAX_LEN, 0);
  std::vector<uint16_t> attention_mask(MAX_LEN * MAX_LEN, BF16_NEG_10000);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < MAX_LEN; j++) {
      if (j <= i) {
        attention_mask[i * MAX_LEN + j] = 0;
      }
    }
  }

  // forward embeding
  bm_memcpy_s2d(bm_handle, inputs_embed_512.device_mem,
                (void *)input_ids.data());
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &inputs_embed_512, 1,
                            &outputs_embed_512, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem,
                (void *)attention_mask.data());
  auto inputs_embed = outputs_embed_512;
  bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(), inputs_block, 3,
                                outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    move2end(past_key[i]);
    move2end(past_value[i]);
  }
  int bytes = inputs_embed.device_mem.size / MAX_LEN;
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
  std::vector<uint16_t> attention_mask(MAX_LEN + 1, 0);
  for (int i = 0; i <= MAX_LEN - token_length; i++) {
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
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t inputs_block[5] = {inputs_embed, next_pid, next_attention,
                                   past_key[i], past_value[i]};
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block, 5, outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
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
  auto input_ids = tk->encode_history(history, MAX_LEN);
  token_length = input_ids.size();
  auto time_1 = std::chrono::system_clock::now();
  int pre_token = 0;
  int token = forward_first(input_ids);
  auto time_2 = std::chrono::system_clock::now();
  std::string result;
  while (token != tk->im_end_id && token_length < MAX_LEN) {
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    auto pre_word = tk->decode(pre_ids);
    auto word = tk->decode(ids);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    std::cout << diff << std::flush;
    if (token_length < MAX_LEN) {
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
  if (token_length >= MAX_LEN) {
    printf(" ......\nWarning: cleanup early history\n");
  }
  // double tht = tokens.size() / (tht_dur.count() * 1e-6);
  printf("\nFTL:%f s, TPS: %f tokens/s\n", ftl_dur.count() * 1e-6, tps);
  history.emplace_back(result);
  if (token_length + 128 >= MAX_LEN) {
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


extern "C" {

struct StringVector {
    const char** data;
    size_t size;
};

struct Slice {
    int data[10000];// be careful of the size
    size_t size;
};

QwenChat* QwenChat_with_device_model(int devid, const char* bmodel_path, const char* tokenizer_path);
QwenChat* QwenChat_with_devices_models(int* devids,int device_len , const char* bmodel_path, const char* tokenizer_path);
void QwenChat_deinit( QwenChat* chat );
StringVector get_history( QwenChat* chat );
void set_history( QwenChat* chat, StringVector history_c );
int predict_first_token_with_str( QwenChat* chat, const char* input_str );
int predict_first_token_with_tokens( QwenChat* chat, int* tokens, int token_len ) ;
int predict_next_token( QwenChat* chat );
void tokens2str( QwenChat* chat, int* tokens, int token_len, char* str );
void token2str_with_pre( QwenChat* chat, int token, char* str );
void str2tokens( QwenChat* chat, StringVector* history_c, Slice* slice_data) ;
int get_token_length( QwenChat* chat );

QwenChat* QwenChat_with_device_model(int devid, const char* bmodel_path, const char* tokenizer_path){
    std::vector<int> devices = {devid};
    QwenChat* chat = new QwenChat();
    chat->init(devices, bmodel_path, tokenizer_path);
    return chat;
}

QwenChat* QwenChat_with_devices_models(int* devids, int device_len , const char* bmodel_path, const char* tokenizer_path){
    std::vector<int> devices;
    for(int i=0;i<device_len;i++){
        devices.push_back(devids[i]);
    }
    QwenChat* chat = new QwenChat();
    chat->init(devices, bmodel_path);
    return chat;
}

void QwenChat_deinit( QwenChat* chat ) { chat->deinit(); delete chat; }

StringVector get_history( QwenChat* chat ) { 
    auto history = chat->history;
    StringVector sv;
    sv.size = history.size();
    sv.data = new const char*[sv.size];
    for(size_t i=0;i<sv.size;i++){
        sv.data[i] = history[i].c_str();
    }
    return sv;
}

void set_history( QwenChat* chat, StringVector history_c ) { 
    auto history = chat->history;
    history.clear();
    for (size_t i=0; i<history_c.size; i++) {
        history.emplace_back( std::string( history_c.data[i] ) );
    }
}

int predict_first_token_with_str( QwenChat* chat, const char* input_str ) { 
    chat->history.emplace_back(std::move(input_str));
    auto input_ids = chat->tk->encode_history(chat->history, MAX_LEN);
    return predict_first_token_with_tokens(chat, input_ids.data(), input_ids.size());
}

int predict_first_token_with_tokens( QwenChat* chat, int* tokens, int token_len ) { 
    // todo : check something 
    std::vector<int> input_ids;
    for(int i=0;i<token_len;i++){
        input_ids.push_back(tokens[i]);
    }
    chat->token_length = input_ids.size();
    int token = chat->forward_first(input_ids);
    chat->token_length ++;
    return token;
}

int predict_next_token( QwenChat* chat ) { 
    // TODO
    int token = chat->forward_next();
    chat->token_length ++;
    return token;
}

void tokens2str( QwenChat* chat, int* tokens, int token_len, char* str ) { 
    std::vector<int> ids;
    for(int i=0;i<token_len;i++){
        ids.push_back(tokens[i]);
    }
    auto word = chat->tk->decode(ids);
    strcpy(str, word.c_str());
}

void str2tokens( QwenChat* chat, StringVector* history_c, Slice* slice_data) { 
    std::vector<std::string> history;
    for(size_t i=0;i<history_c->size;i++){
        history.emplace_back( std::string( history_c->data[i] ) );
    }
    auto input_ids = chat->tk->encode_history(history, MAX_LEN);
    slice_data->size = input_ids.size();
    for(size_t i=0;i<slice_data->size;i++){
        slice_data->data[i] = input_ids[i];
    }
}

void token2str_with_pre( QwenChat* chat, int token, char* str ) { 
    int pre_token = 0;
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    pre_word = chat->tk->decode(pre_ids);
    word     = chat->tk->decode(ids);
    std::string diff = word.substr(pre_word.size());
    strcpy(str, diff.c_str());
}

int get_token_length( QwenChat* chat ) { 
    return chat->token_length;
}

int get_eos( QwenChat* chat ) { 
    return chat->tk->im_end_id;
}

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
    const char* bmodel_path = "/workspace/aa/qwen-7b_int4.bmodel";
    const char* tokenizer_path = "/workspace/aa/Qwen-TPU/webdemo/qwen.tiktoken";
    auto llm = QwenChat_with_device_model(0, bmodel_path, tokenizer_path);
    const char* input_str   = "你能干啥";
    int eos  = get_eos(llm);
    printf("eos: %d\n", eos);
    char str[1000];
    int token = predict_first_token_with_str(llm, input_str);
    printf("token: %d\n", token);
    tokens2str(llm, &token, 1, str);
    printf("cur str: %s\n", str);
    printf("cur_length: %d\n", get_token_length(llm));
    while(true){
        if(token == eos && get_token_length(llm) < MAX_LEN){
            break;
        }
        int next_token = predict_next_token(llm);
        printf("next_token: %d\n", next_token);
        tokens2str(llm, &next_token, 1, str);
        printf("cur str: %s\n", str);
        printf("cur_length: %d\n", get_token_length(llm));
        token = next_token;
    }
}
