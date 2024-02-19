import ctypes 
import numpy as np 

MAX_TOKENS = 10000;
MAX_LEN    = 512;

def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)

def make2_c_int_list(my_list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point):
    return ctypes.string_at(char_point).decode('utf-8')

def make_np2c(np_array):
    if np_array.flags['CONTIGUOUS'] == False:
        # info users
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.c_void_p)

def str2char_point(string):
    return ctypes.c_char_p(string.encode('utf-8'))


class StringVector(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_char_p)),
        ('size', ctypes.c_size_t),
    ]

class Slice(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_int * 10000),
        ('size', ctypes.c_size_t),
    ]

lib = ctypes.cdll.LoadLibrary('./libqwenc.so')

lib.QwenChat_with_device_model.restype  = ctypes.c_void_p
lib.QwenChat_with_device_model.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
def QwenChat_with_device_model(devid, bmodel_path, tokenizer_path) -> ctypes.c_void_p:
    """
    QwenChat* QwenChat_with_device_model(int devid, const char* bmodel_path, const char* tokenizer_path);
    :param devid: 	ctypes.c_int
    :param bmodel_path: 	ctypes.c_char_p
    :param tokenizer_path: 	ctypes.c_char_p
    """
    return lib.QwenChat_with_device_model(ctypes.c_int(devid), str2char_point(bmodel_path), str2char_point(tokenizer_path))

lib.QwenChat_with_devices_models.restype  = ctypes.c_void_p
lib.QwenChat_with_devices_models.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
def QwenChat_with_devices_models(devids, device_len, bmodel_path, tokenizer_path) -> ctypes.c_void_p:
    """
    QwenChat* QwenChat_with_devices_models(int* devids,int device_len , const char* bmodel_path, const char* tokenizer_path);
    :param devids: 	ctypes.POINTER(ctypes.c_int)
    :param device_len: 	ctypes.c_int
    :param bmodel_path: 	ctypes.c_char_p
    :param tokenizer_path: 	ctypes.c_char_p
    """
    return lib.QwenChat_with_devices_models(make2_c_int_list(devids), ctypes.c_int(device_len), str2char_point(bmodel_path), str2char_point(tokenizer_path))

lib.QwenChat_deinit.restype  = None
lib.QwenChat_deinit.argtypes = [ctypes.c_void_p]
def QwenChat_deinit(chat) -> None:
    """
    void QwenChat_deinit( QwenChat* chat );
    :param chat: 	ctypes.c_void_p
    """
    return lib.QwenChat_deinit(chat)

lib.get_history.restype  = StringVector
lib.get_history.argtypes = [ctypes.c_void_p]
def get_history(chat) -> StringVector:
    """
    StringVector get_history( QwenChat* chat );
    :param chat: 	ctypes.c_void_p
    """
    return lib.get_history(chat)

lib.set_history.restype  = None
lib.set_history.argtypes = [ctypes.c_void_p, StringVector]
def set_history(chat, history_c) -> None:
    """
    void set_history( QwenChat* chat, StringVector history_c );
    :param chat: 	ctypes.c_void_p
    :param history_c: 	StringVector
    """
    return lib.set_history(chat, history_c)

lib.predict_first_token_with_str.restype  = ctypes.c_int
lib.predict_first_token_with_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
def predict_first_token_with_str(chat, input_str) -> ctypes.c_int:
    """
    int predict_first_token_with_str( QwenChat* chat, const char* input_str );
    :param chat: 	ctypes.c_void_p
    :param input_str: 	ctypes.c_char_p
    """
    return lib.predict_first_token_with_str(chat, str2char_point(input_str))

lib.predict_first_token_with_tokens.restype  = ctypes.c_int
lib.predict_first_token_with_tokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
def predict_first_token_with_tokens(chat, tokens) -> ctypes.c_int:
    """
    modified by zwy
    int predict_first_token_with_tokens( QwenChat* chat, int* tokens, int token_len ) ;
    :param chat: 	ctypes.c_void_p
    :param tokens: 	ctypes.POINTER(ctypes.c_int)
    :param token_len: 	ctypes.c_int
    """
    token_len = len(tokens)
    return lib.predict_first_token_with_tokens(chat, make2_c_int_list(tokens), ctypes.c_int(token_len))

lib.predict_next_token.restype  = ctypes.c_int
lib.predict_next_token.argtypes = [ctypes.c_void_p]
def predict_next_token(chat) -> ctypes.c_int:
    """
    int predict_next_token( QwenChat* chat );
    :param chat: 	ctypes.c_void_p
    """
    return lib.predict_next_token(chat)

lib.tokens2str.restype  = None
lib.tokens2str.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
def tokens2str(chat, tokens):
    """
    modifed by zwy 
    void tokens2str( QwenChat* chat, int* tokens, int token_len, char* str );
    :param chat: 	ctypes.c_void_p
    :param tokens: 	ctypes.POINTER(ctypes.c_int)
    :param token_len: 	ctypes.c_int
    :param str: 	ctypes.c_char_p
    """
    token_len = len(tokens)
    str = ctypes.create_string_buffer(MAX_TOKENS)
    lib.tokens2str(chat, make2_c_int_list(tokens), ctypes.c_int(token_len), str)
    return str.value.decode('utf-8', 'ignore')

lib.token2str_with_pre.restype  = None
lib.token2str_with_pre.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
def token2str_with_pre(chat, token) -> None:
    """
    modifed by zwy
    void token2str_with_pre( QwenChat* chat, int token, char* str );
    :param chat: 	ctypes.c_void_p
    :param token: 	ctypes.c_int
    :param str: 	ctypes.c_char_p
    """
    str = ctypes.create_string_buffer(MAX_TOKENS)
    lib.token2str_with_pre(chat, ctypes.c_int(token), str)
    return str.value.decode('utf-8', 'ignore')

lib.str2tokens.restype  = None
lib.str2tokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(StringVector), ctypes.POINTER(Slice)]
def str2tokens(chat, history_c, slice_data) -> None:
    """
    void str2tokens( QwenChat* chat, StringVector* history_c, Slice* slice_data) ;
    :param chat: 	ctypes.c_void_p
    :param history_c: 	ctypes.POINTER(StringVector)
    :param slice_data: 	ctypes.POINTER(Slice)
    """
    return lib.str2tokens(chat, ctypes.byref(history_c), ctypes.byref(slice_data))

lib.get_token_length.restype  = ctypes.c_int
lib.get_token_length.argtypes = [ctypes.c_void_p]
def get_token_length(chat) -> ctypes.c_int:
    """
    int get_token_length( QwenChat* chat );
    :param chat: 	ctypes.c_void_p
    """
    return lib.get_token_length(chat)
version='2023-12-18-16-00-15'

lib.get_eos.restype  = ctypes.c_int
lib.get_eos.argtypes = [ctypes.c_void_p]


class QwenChat:
    
    def __init__(self, devid=0, bmodel_path="/data/Qwen/qwen-7b_int4_fast.bmodel", tokenizer_path="/data/Qwen/qwen.tiktoken"):
        self.chat       = QwenChat_with_device_model(devid, bmodel_path, tokenizer_path)
        self.history    = []
        self.history_c  = StringVector()
        self.tokens     = Slice()
        self.tokens.size = 0
        self.eos        = lib.get_eos(self.chat)
        self.tokenizer_path = tokenizer_path
        self.bmodel_path = bmodel_path
        self.cur_token = None

    def __del__(self):
        pass
    
    @property
    def token_len(self):
        return get_token_length(self.chat)
        
    def history_to_c(self):
        self.history_c.size = len(self.history)
        c_strings = [ctypes.c_char_p(string.encode()) for string in self.history]
        self.history_c.data = (ctypes.c_char_p * len(c_strings))(*c_strings)
    
    def historyc_to_history(self):
        # decoder c string vector
        history = []
        size    = self.history_c.size
        data    = self.history_c.data
        for i in range(size):
            history.append(char_point_2_str(data[i]))
        return history
    
    def add_history(self, content:str):
        
        pass
    
    def predict_next_token(self):
        pass
    
    def handle_token_in_single_conversation(self):
        
        pass
    
    def predict(self, context:str):
        # 如果当前的token length 小于多少就得减少历史记录 
        # 判断当前的context 
        # 假设都符合 
        # 是否降低hisoty todo by 路程 
        self.history.append(context)
        self.history_to_c()
        str2tokens(self.chat, self.history_c, self.tokens)
        first_token = predict_first_token_with_tokens(self.chat, self.tokens.data[:self.tokens.size])
        first_str   = token2str_with_pre(self.chat, first_token)
        print(first_str, end="")
        res = first_str
        self.cur_token = first_token
        while True:
            if self.cur_token == self.eos:
                break
            if self.token_len >= MAX_LEN:
                print("......\n Warning: cleanup early history")
                break
            next_token = predict_next_token(self.chat)
            next_str   = token2str_with_pre(self.chat, next_token)
            print(next_str, end='',flush=True)
            res += next_str
            self.cur_token = next_token
        self.history.append(res)
        return res

    def predict_no_state(self, question: str, history: list):
        # question: "你多大了？" history: ["你是谁？", "我是Qwen。"]
        self.history = history
        self.history.append(question)
        self.history_to_c()
        str2tokens(self.chat, self.history_c, self.tokens)

        first_token = predict_first_token_with_tokens(self.chat, self.tokens.data[:self.tokens.size])
        first_str   = token2str_with_pre(self.chat, first_token)

        res = first_str
        self.cur_token = first_token
        
        while True:
            if self.cur_token == self.eos:
                break
            if self.token_len >= MAX_LEN:
                print("......\n Warning: cleanup early history")
                break
            next_token = predict_next_token(self.chat)
            next_str   = token2str_with_pre(self.chat, next_token)
            # print(next_str, end='',flush=True)
            res += next_str
            self.cur_token = next_token

            yield {
                "data": res
            }

if __name__=="__main__":
    qwen = QwenChat()
    while True:
        try:
            context = input("Qwen: ")
            qwen.predict(context)
            print("\n")
        except KeyboardInterrupt:
            break
    pass
