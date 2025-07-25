import asyncio
import websockets
import numpy as np
import whisper
import io

model = whisper.load_model("base")  # 你也可以用 tiny, small, medium 等

# 保存每个连接的缓冲音频
connections = {}

async def handler(websocket, path):
    print(f"Client connected: {websocket.remote_address}")
    buffer = bytes()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                buffer += message

                # 一次性识别逻辑（可替换为流式识别）
                if len(buffer) >= 32000:  # 至少1秒音频（16k采样 * 2字节）
                    float_array = np.frombuffer(buffer, dtype=np.float32)
                    audio = whisper.pad_or_trim(float_array)

                    # Whisper 模型预处理
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    result = model.transcribe(audio)

                    # 发送识别结果给客户端
                    text = result["text"]
                    await websocket.send(text.strip())

                    # 清空缓冲（也可保留部分历史用于上下文识别）
                    buffer = bytes()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 19463, max_size=2**22):
        print("WebSocket server listening on ws://0.0.0.0:19463/recognition")
        await asyncio.Future()  # 永不结束

asyncio.run(main())
