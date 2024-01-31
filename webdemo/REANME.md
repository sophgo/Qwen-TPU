### HTTP API

#### 启动方法
`python api.py --bmodel_path bmodel_path --tokenizer_path tokenizer_path`

#### 使用说明
- 地址`http://127.0.0.1:8000/chat/completions`
- 方法`POST`
- 请求体
	- question: 当前问题
	- history: 历史聊天记录，为一维列表，索引位0，2，4，··· 存储用户问题，索引位1，3，5，··· 存储AI回答
	- stream: 是否采用流式输出
- 请求体示例 
```
	{
		"question": "南京的呐？",
		"history": ["北京的面积有多大？", "北京市总面积为16410平方千米。"],
		"stream": true
	}
```
- 返回体示例
```
{
	"data": "南京市总面积为6587平方千米。"
}
```


### Gradio Demo

#### 启动方法
`python web_demo.py --bmodel_path bmodel_path --tokenizer_path tokenizer_path`

#### 使用说明
- 地址`http://127.0.0.1:7860/`