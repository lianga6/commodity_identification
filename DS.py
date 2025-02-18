#非流式（已经成功）

import json
from vosk import Model, KaldiRecognizer
import pyaudio
import keyboard
import wave
from openai import OpenAI
import pyttsx3
import time

# 加载模型
model = Model(r"vosk-model-small-cn-0.22")

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-qVU1PGkwEf8IXBiJybDvbV6NliS5TYBSJJJhY70SvdY1csNL"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://ai.forestsx.top/v1")

# 录音配置
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

def record_audio():
    """录制用户语音"""
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("开始录音...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("录音结束")
    stream.stop_stream()
    stream.close()
    return b''.join(frames)

def save_audio_to_file(audio_data, filename="user_input.wav"):
    """将录音保存为文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

def send_to_deepseek(messages):
    """调用DeepSeek API"""
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=messages
    )
    return response.choices[0].message

def text_to_speech(text):
    """语音播报"""
    print(f"助手: {text}")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speech_to_text(audio_file="user_input.wav"):
    """语音转文字"""
    wf = wave.open(audio_file, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text += result.get("text", "")
    return text

def main():
    print("语音助手已启动，按下 'v' 开始说话，按下 'c' 重置对话，按下 'q' 退出程序")
    messages = [{"role": "system", "content": "你的名字叫超市哥，是伟大的亮哥超市的智能助手，用于服务顾客，与顾客交流记得要言简意赅，不要太啰嗦"},]
    while True:
        time.sleep(0.1)  # 降低CPU占用
        
        if keyboard.is_pressed('v'):
            # 防抖处理
            while keyboard.is_pressed('v'):
                time.sleep(0.1)
                
            audio_data = record_audio()
            save_audio_to_file(audio_data)
            user_input = speech_to_text("user_input.wav")
            print(f"用户: {user_input}")
            
            messages.append({"role": "user", "content": user_input})
            response = send_to_deepseek(messages)
            messages.append(response)
            text_to_speech(response.content)

        elif keyboard.is_pressed('c'):
            # 防抖处理
            while keyboard.is_pressed('c'):
                time.sleep(0.1)
                
            messages = []
            print("对话已重置，语音助手初始化完成")

        elif keyboard.is_pressed('q'):
            print("正在退出程序...")
            break

    audio.terminate()  # 释放音频资源
    print("程序已退出")

if __name__ == "__main__":
    main()
