#流式，静音检测
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import pyaudio
import keyboard
import wave
from openai import OpenAI
import time
import subprocess
from dashscope.audio.tts import SpeechSynthesizer
from pygame import mixer
import pygame
import dashscope
import threading
import tempfile

# ----------------- 系统级音频配置 -----------------
subprocess.run(["amixer", "cset", "numid=3", "1"])
subprocess.run(["pactl", "set-default-sink", "0"])

# 初始化音频混合器
mixer.init()
pygame.init()

# ----------------- 音频设备初始化 -----------------
def get_audio_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        devices.append({
            "index": i,
            "name": dev["name"],
            "maxInputChannels": dev["maxInputChannels"],
            "maxOutputChannels": dev["maxOutputChannels"],
            "defaultSampleRate": dev["defaultSampleRate"]
        })
    p.terminate()
    return devices

print("=== 可用音频设备 ===")
for dev in get_audio_devices():
    print(f"Index {dev['index']}: {dev['name']} | 输入通道: {dev['maxInputChannels']} | 输出通道: {dev['maxOutputChannels']}")

# ----------------- 配置部分 -----------------
model = Model(r"vosk-model-small-cn-0.22")
DEEPSEEK_API_KEY = "sk-qVU1PGkwEf8IXBiJybDvbV6NliS5TYBSJJJhY70SvdY1csNL"
DASHSCOPE_API_KEY = "sk-7a75660a7f1c46e8bcf9d7f06ea80be0"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://ai.forestsx.top/v1")
dashscope.api_key = DASHSCOPE_API_KEY

# 新增静音检测参数
CHUNK = 1024
SILENCE_THRESHOLD = 500      # 静音阈值（根据环境调整）
SILENCE_DURATION = 1.5       # 静音持续时间（秒）
MAX_RECORD_TIME = 30         # 最大录音时长（秒）

# ----------------- 音频设备选择 -----------------
def select_input_device():
    for dev in get_audio_devices():
        if "USB" in dev["name"] and dev["maxInputChannels"] > 0:
            print(f"选择输入设备: {dev['name']}")
            return dev["index"]
    raise Exception("未找到USB麦克风")

def select_output_device():
    for dev in get_audio_devices():
        if "bcm2835" in dev["name"] and dev["maxOutputChannels"] > 0:
            print(f"选择输出设备: {dev['name']}")
            return dev["index"]
    raise Exception("未找到3.5mm输出设备")

# ----------------- 改进后的核心功能 -----------------
class AudioSystem:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_device = select_input_device()
        self.output_device = select_output_device()
        self.sample_rate = self.detect_sample_rate()
        self.play_lock = threading.Lock()

    def detect_sample_rate(self):
        test_rates = [16000, 44100, 48000]
        for rate in test_rates:
            try:
                test_stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=self.input_device,
                    frames_per_buffer=CHUNK,
                    start=False
                )
                test_stream.close()
                print(f"使用采样率: {rate}Hz")
                return rate
            except:
                continue
        raise Exception("无可用采样率")

    def record_audio(self):
        """智能录音功能（带静音检测）"""
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=CHUNK
        )
        
        print("\033[90m检测到语音输入，开始录音...\033[0m")  # 灰色提示
        frames = []
        last_sound_time = time.time()
        start_time = time.time()
        is_speaking = False

        while True:
            # 超时保护
            if time.time() - start_time > MAX_RECORD_TIME:
                print("\033[90m达到最大录音时长\033[0m")
                break

            data = stream.read(CHUNK)
            frames.append(data)
            
            # 计算当前音频块的能量值
            audio_data = np.frombuffer(data, dtype=np.int16)
            current_energy = np.abs(audio_data).max()

            # 语音活动检测逻辑
            if current_energy > SILENCE_THRESHOLD:
                if not is_speaking:
                    print("\033[90m检测到语音开始\033[0m")
                    is_speaking = True
                last_sound_time = time.time()
            else:
                if is_speaking and (time.time() - last_sound_time) > SILENCE_DURATION:
                    print("\033[90m检测到语音结束\033[0m")
                    break

        stream.stop_stream()
        stream.close()
        print(f"\033[90m录音结束，时长：{time.time()-start_time:.2f}秒\033[0m")
        return b''.join(frames)

    def _synthesize(self, text):
        try:
            result = SpeechSynthesizer.call(
                model='sambert-zhimiao-emo-v1',
                text=text,
                sample_rate=48000,
                format='wav'
            )
            return result.get_audio_data()
        except Exception as e:
            print(f"\033[91m语音合成失败: {str(e)}\033[0m")
            return None

    def text_to_speech(self, text):
        """保持原有线程播放功能"""
        if len(text) == 0:
            return
        
        def play_task():
            with self.play_lock:
                audio_data = self._synthesize(text)
                if audio_data:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                        temp_file.write(audio_data)
                        temp_file.flush()
                        
                        mixer.music.load(temp_file.name)
                        mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
        
        threading.Thread(target=play_task).start()

# ----------------- 主程序 ----------------- 
def main():
    audio_sys = AudioSystem()
    
    def save_audio(audio_data):
        with wave.open("user_input.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio_sys.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(audio_sys.sample_rate)
            wf.writeframes(audio_data)

    def speech_to_text():
        rec = KaldiRecognizer(model, audio_sys.sample_rate)
        with wave.open("user_input.wav", 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
            return json.loads(rec.FinalResult())["text"]

    messages = [{
        "role": "system",
        "content": "你的名字叫超市哥，是伟大的亮哥超市的智能助手，用于服务顾客，与顾客交流记得要言简意赅，不要太啰嗦"
    }]

    print("\033[95m语音助手已启动，按下 'v' 开始说话，按下 'c' 重置对话，按下 'q' 退出程序\033[0m")  # 紫色提示

    while True:
        time.sleep(0.1)
        
        if keyboard.is_pressed('v'):
            while keyboard.is_pressed('v'): time.sleep(0.1)
            
            audio_data = audio_sys.record_audio()
            save_audio(audio_data)
            
            user_input = speech_to_text()
            print(f"\033[94m用户: {user_input}\033[0m")  # 蓝色显示用户输入
            
            messages.append({"role": "user", "content": user_input})
            
            # 流式响应处理
            buffer = ""
            full_response = ""
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    buffer += content
                    full_response += content
                    
                    # 中文句号分割处理
                    while "。" in buffer:
                        pos = buffer.find("。")
                        sentence = buffer[:pos+1].strip()
                        buffer = buffer[pos+1:].lstrip()
                        
                        if sentence:
                            audio_sys.text_to_speech(sentence)
            
            # 处理剩余内容
            if buffer.strip():
                audio_sys.text_to_speech(buffer.strip())

            print(f"\033[92m助手: {full_response}\033[0m")  # 绿色显示助手回复
            
            messages.append({"role": "assistant", "content": full_response})

        elif keyboard.is_pressed('c'):
            while keyboard.is_pressed('c'): time.sleep(0.1)
            messages = messages[:1]
            print("\033[93m对话已重置\033[0m")  # 黄色显示重置提示

        elif keyboard.is_pressed('q'):
            print("\033[91m退出中...\033[0m")  # 红色显示退出提示
            audio_sys.p.terminate()
            break

if __name__ == "__main__":
    main()
