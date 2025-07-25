import asyncio
import websockets
import numpy as np
from funasr import AutoModel
import queue

from sympy.physics.units import length

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
        "model_dir": "models/SenseVoiceSmall",
        "output_file": "tmp/"
    }
)
audio_queue = asyncio.Queue()
vad_queue = asyncio.Queue()
connected_websocket = None
async def handle_client(websocket, path):
    global connected_websocket
    connected_websocket = websocket  # ✅ 保存当前连接
    print("✅ 客户端已连接")
    buffer = bytearray()
    frame_samples = 512
    frame_bytes = frame_samples * 2  # int16 = 2 bytes
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                buffer.extend(message)
                # 尽可能多处理完整帧（512 samples）
                while len(buffer) >= frame_bytes:
                    if len(buffer) < frame_bytes:
                        break  # 丢弃不完整帧
                    chunk = buffer[:frame_bytes]
                    buffer = buffer[frame_bytes:]
                    # 校验帧长度
                    if len(chunk) != frame_bytes:
                        print(f"⚠️ 无效帧长度: {len(chunk)} bytes")
                        continue
                    try:
                        # 先以 int16 读取，再转为 float32 [-1, 1]
                        int_audio = np.frombuffer(chunk, dtype=np.int16)
                        float_audio = int_audio.astype(np.float32) / 32768.0
                        if float_audio.shape[0] != frame_samples:
                            print(f"⚠️ 错误帧长度：{float_audio.shape[0]}，跳过")
                            continue

                        # 调用 VAD
                        vad_statue = vad.is_unity_vad(float_audio)
                        # 推入异步处理队列（注意 asyncio.Queue）
                        await vad_queue.put({
                            "voice": chunk,
                            "vad_statue": vad_statue
                        })
                    except Exception as e:
                        print(f"❌ VAD 处理出错: {e}")
    except websockets.exceptions.ConnectionClosed:
        print("❌ 客户端断开连接")
in_speech = False
current_voice_chunks = []
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
    async with websockets.serve(handle_client, "0.0.0.0", 19463):
        print("🚀 WebSocket 语音识别服务运行中：ws://0.0.0.0:19463")
        await asyncio.Future()


asyncio.run(main())
