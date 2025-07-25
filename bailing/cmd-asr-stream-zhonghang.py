import asyncio
import websockets
import numpy as np
from funasr import AutoModel
import queue

from sympy.physics.units import length
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bailing import vad,asr
vad = vad.create_instance(
    "SileroVAD",
    {
        "sampling_rate": 16000,
        "threshold": 0.2,
        "min_silence_duration_ms": 500
    }
)
asr = asr.create_instance(
    "FunASR",
    {
        "model": None,  # ✅ 明确告诉 FunASR 不要从 huggingface 拉模型
        "model_dir": "models/SenseVoiceSmall",
        "output_file": "tmp/"
    }
)
from scipy.signal import resample
audio_queue = asyncio.Queue()
vad_queue = asyncio.Queue()
connected_websocket = None
ORIG_SR = 48000
TARGET_SR = 16000
VAD_FRAME_SAMPLES = 512
INPUT_FRAME_SAMPLES = int(VAD_FRAME_SAMPLES * ORIG_SR / TARGET_SR)  # 1536
INPUT_FRAME_BYTES = INPUT_FRAME_SAMPLES * 2  # int16 = 2 bytes
def resample_audio(int16_audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    float_audio = int16_audio.astype(np.float32) / 32768.0  # int16 PCM -> float32 [-1, 1]
    duration = len(float_audio) / orig_sr
    target_len = int(duration * target_sr)
    resampled = resample(float_audio, target_len)  # scipy.signal.resample
    return resampled.astype(np.float32)

async def handle_client(websocket, path):
    buffer = bytearray()
    async for message in websocket:
        if isinstance(message, bytes):
            buffer.extend(message)

            # 尽可能多处理完整的 1536 个采样点的帧
            while len(buffer) >= INPUT_FRAME_BYTES:
                chunk = buffer[:INPUT_FRAME_BYTES]
                buffer = buffer[INPUT_FRAME_BYTES:]

                try:
                    # 1. 解码为 int16
                    int_audio = np.frombuffer(chunk, dtype=np.int16)

                    # 2. 转 float32 [-1, 1]
                    float_audio = int_audio.astype(np.float32) / 32768.0

                    # 3. 重采样 48000 -> 16000
                    resampled = resample(float_audio, VAD_FRAME_SAMPLES)  # 得到 512 float32

                    # 4. 校验
                    if resampled.shape[0] != VAD_FRAME_SAMPLES:
                        print(f"⚠️ 重采样后帧错误: {resampled.shape}")
                        continue

                    # 5. 推入 VAD（用你的原始方法）
                    vad_statue = vad.is_unity_vad(resampled)
                    int16_resampled = (np.clip(resampled, -1, 1) * 32767).astype(np.int16)
                    # 6. 推入队列
                    await vad_queue.put({
                        "voice": int16_resampled.tobytes(),  # 你可以保留原始数据
                        "vad_statue": vad_statue
                    })
                except Exception as e:
                    print(f"❌ VAD 处理异常: {e}")

in_speech = False
current_voice_chunks = []
API_WENDA="http://127.0.0.1:9885/chatgpt/api/getWendaContent/zhonghang/active"
API_MESSAGE="http://127.0.0.1:5010/api/push"
REDIS_WAITING_KEYS="ceyan:questionWaiting:python"
import requests
import json
from redis_global import redis_manager
#处理vad队列
async def process_vad_queue():
    global in_speech, current_voice_chunks

    while True:
        if not vad_queue.empty():
            data = await vad_queue.get()
            voice_data = data["voice"]
            vad_statue = data["vad_statue"]

            # ❗️跳过无效帧
            if voice_data is None or not isinstance(voice_data, (bytes, bytearray)):
                print(f"❌ 跳过无效 voice_data 类型: {type(voice_data)}")
                continue

            # ✅ VAD 识别开始
            if vad_statue and "start" in vad_statue:
                print("🟢 VAD 检测到语音开始")
                in_speech = True
                current_voice_chunks = []
                current_voice_chunks.append(bytes(voice_data))  # 转 bytes

            # ✅ 语音过程中持续收集数据
            elif in_speech:
                current_voice_chunks.append(bytes(voice_data))

                if vad_statue and "end" in vad_statue:
                    print("🔴 VAD 检测到语音结束")
                    in_speech = False
                    try:
                        print(f"📦 current_voice_chunks 类型: {[type(chunk) for chunk in current_voice_chunks]}")
                        full_voice = b"".join(current_voice_chunks)
                        current_voice_chunks = []
                        text, tmpfile = asr.recognizer_unity(full_voice)
                        if not text or not text.strip():
                            print("识别结果为空，跳过处理")
                            continue
                        print(f"📝 识别结果: {text}")
                        # 参数 {"type":"easy_wav2lip","video_path":{"path":"ceyan.mp4","format":"mp4"},"audio_path":"baidu_9.wav","insert_index":0,"interrupt":true}
                        json_msg = {
                            "type": "easy_wav2lip",
                            "text": text,
                            "insert_index": 0,
                            "interrupt": True
                        }
                        # 将内容推送api接口到前端
                        feature_recorded = redis_manager.get_key_value(REDIS_WAITING_KEYS)
                        print(f'feature_recorded: {feature_recorded}')
                        # 判断feature_recorded是否存在
                        if feature_recorded is not None:
                            requests.post(API_MESSAGE, json=json_msg)
                        # 调用Java的api接口
                        params = {
                            "prompt": text,
                        }
                        # 调用api接口 API_WENDA
                        java_response = requests.post(API_WENDA, json=params)
                        java_response.raise_for_status()
                        java_result = java_response.json()  # 假设返回的是JSON
                        print("Java API response:", java_result)
                        # 提取JSON内容
                        json_content = java_result.get("resultStr")
                        # 将内容推送api接口到前端
                        feature_recorded = redis_manager.get_key_value(REDIS_WAITING_KEYS)
                        if json_content is not None and feature_recorded is not None and java_result.get("code") == 200:
                            print("Java API response:", java_result.get("code"))
                            parsed_content = json.loads(json_content)
                            text = parsed_content.get("text", "")  # 提取内部 text 字段
                            json_msg = {
                                "type": "easy_wav2lip",
                                "text": text,  # 从JSON中提取文本
                                "insert_index": 0,
                                "interrupt": True
                            }
                            print(f'Java API response: {java_result}')
                            requests.post(API_MESSAGE, json=json_msg)
                        #通过websock发送到unity
                        # ✅ 通过 websocket 发回 Unity 客户端
                        if connected_websocket and connected_websocket.open:
                            try:
                                #如果text长度小于2 则不发送
                                if len(text) < 3:
                                    print("⚠️ 识别结果过短，不发送")
                                    continue
                                message ="\n"  # 添加换行符用于 Unity 判断结束
                                await connected_websocket.send(text )  # 直接发送识别结果文本
                                #结束
                                await connected_websocket.send(message)  # 直接发送识别结果文本
                                print("📤 识别结果已发送到客户端")
                            except Exception as e:
                                print(f"❌ 发送识别结果失败: {e}")
                        else:
                            print("⚠️ 无可用的客户端连接，无法发送识别结果")

                    except Exception as e:
                        print(f"❌ 识别错误: {e}")
        await asyncio.sleep(0.01)

async def main():
    # 运行处理VAD队列的任务
    asyncio.create_task(process_vad_queue())
    async with websockets.serve(handle_client, "0.0.0.0", 8012):
        print("🚀 WebSocket 语音识别服务运行中：ws://0.0.0.0:8012")
        await asyncio.Future()


asyncio.run(main())
