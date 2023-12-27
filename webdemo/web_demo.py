import time
import gradio as gr
import mdtex2html
from qwenc import QwenChat
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--bmodel_path', type=str, default = "/workspace/aa/qwen-7b_int4.bmodel")
parser.add_argument('--tokenizer_path', type=str, default= "/workspace/aa/Qwen-TPU/webdemo/qwen.tiktoken")
args = parser.parse_args()


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

glm = QwenChat(bmodel_path=args.bmodel_path, tokenizer_path=args.tokenizer_path)



def predict(input, chatbot, history):
    spilt = None

    # 当前用户输入和历史聊天记录超过320个字时，对超过的历史聊天记录进行截断
    if len(history) > 0:
        for i in range(1, len(history) + 1):
            prompt = ["{} {} ".format(x[0], x[1]) for i, x in enumerate(history[-i:])]
            final_prompt = "".join(prompt) + input
            if len(final_prompt) > 320:
                spilt = -(i - 1)
                break

    if spilt is not None:
        if spilt == 0:
            history = []
        else:
            history = history[spilt:]

    times = []

    chatbot.append((input, ""))
    history1dim = [history[index // 2][index % 2] for index in range(2 * len(history))]
    history.append((input, ""))
    print("input", input)
    print("history1dim", history1dim)
    for response in glm.predict_no_state(input, history1dim):
        chatbot[-1] = (input, response['data'])
        history[-1] = (input, response['data'])
        times.append(time.time())
        yield chatbot, history
    for i in range(len(times) - 1):
        times[i] = times[i+1] - times[i]
    times = times[:-1]
    average = 1 / (sum(times) / len(times))
    print(f"token count: {len(times)}, speed: {format(average, '.4f')} tokens/s.")



def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">QWen TPU</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")

    history_chat = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, history_chat],
                    [chatbot, history_chat], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history_chat], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860)
