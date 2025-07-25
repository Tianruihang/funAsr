import queue
import argparse
import logging
import queue
import threading
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from flask_caching import Cache
from flask_cors import CORS
import requests
from flask import Flask
from fastapi.middleware.cors import CORSMiddleware
from flask_caching import Cache
import recorder
import vad
import  asr
from utils import read_config

# 创建 FastAPI 应用
app = Flask(__name__)

app.config["CACHE_TYPE"] = "RedisCache"
app.config["CACHE_REDIS_HOST"] = "127.0.0.1"  # Redis 服务器地址
app.config["CACHE_REDIS_PORT"] = 6379       # Redis 端口
app.config["CACHE_REDIS_DB"] = 2      # 使用的数据库编号
app.config["CACHE_DEFAULT_TIMEOUT"] = 300    # 默认缓存5分钟
cache = Cache(app)



# 添加CORS中间件解决跨域问题
#批量添加跨域
CORS(app, resources={
    r"/vop": {"origins": "*"},
    r"/vop/v2": {"origins": "*"},
})

#调用本地接口 /chatgpt/api/getWendaContent/v2
API_WENDA="http://127.0.0.1:9885/chatgpt/api/getWendaContent/ceyan"
API_MESSAGE="http://127.0.0.1:8091/show/message"
REDIS_WAITING_KEYS="ceyan:questionWaiting:python"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='asr_stream.log',  # 日志文件名
)
logger = logging.getLogger(__name__)

# 由于deepseek工具调用不太准，经常会输出到content，所以显示指明参数
sys_prompt = """
# 角色定义

"""

class Robot(ABC):
    def __init__(self, config_file):
        config = read_config(config_file)
        self.audio_queue = queue.Queue()

        self.recorder = recorder.create_instance(
            config["selected_module"]["Recorder"],
            config["Recorder"][config["selected_module"]["Recorder"]]
        )


        self.vad = vad.create_instance(
            config["selected_module"]["VAD"],
            config["VAD"][config["selected_module"]["VAD"]]
        )

        self.vad_queue = queue.Queue()

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)

        self.vad_start = True

        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        # 线程锁
        self.chat_lock = False

        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )

        # 事件用于控制程序退出
        self.stop_event = threading.Event()

        self.callback = None

        self.speech = []

        self.task_queue = queue.Queue()

        self.start_task_mode = config.get("StartTaskMode")

    def listen_dialogue(self, callback):
        self.callback = callback

    def _stream_vad(self):
        def vad_thread():
            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get()
                    vad_statue = self.vad.is_vad(data)
                    self.vad_queue.put({"voice": data, "vad_statue": vad_statue})
                except Exception as e:
                    logger.error(f"VAD 处理出错: {e}")
        consumer_audio = threading.Thread(target=vad_thread, daemon=True)
        consumer_audio.start()

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info("Shutting down Robot...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.recorder.stop_recording()
        logger.info("Shutdown complete.")

    def start_recording_and_vad(self):
        # 开始监听语音流
        self.recorder.start_recording(self.audio_queue)
        logger.info("Started recording.")
        # vad 实时识别
        self._stream_vad()

    def _duplex(self):
        # 处理识别结果
        data = self.vad_queue.get()
        # 识别到vad开始
        if self.vad_start:
            self.speech.append(data)
        vad_status = data.get("vad_statue")
        """ 语音唤醒
        if time.time() - self.start_time>=60:
            self.silence_status = True

        if self.silence_status:
            return
        """
        if vad_status is None:
            return
        if "start" in vad_status:
            if self.chat_lock is True:  # 正在播放，打断场景
                if self.INTERRUPT:
                    self.chat_lock = False
                    self.vad_start = True
                    self.speech.append(data)
                else:
                    return
            else:  # 没有播放，正常
                self.vad_start = True
                self.speech.append(data)
        elif "end" in vad_status and len(self.speech) > 0:
            try:
                logger.debug(f"语音包的长度：{len(self.speech)}")
                self.vad_start = False
                voice_data = [d["voice"] for d in self.speech]
                text, tmpfile = self.asr.recognizer(voice_data)
                self.speech = []
            except Exception as e:
                self.vad_start = False
                self.speech = []
                logger.error(f"ASR识别出错: {e}")
                return
            if not text.strip():
                logger.debug("识别结果为空，跳过处理。")
                return
            logger.debug(f"ASR识别结果: {text}")
            print(f'识别结果: {text}')
            # 调用Java的api接口
            params = {
                "prompt": text,
            }
            #调用api接口 API_WENDA
            java_response = requests.post(API_WENDA, json=params)
            java_response.raise_for_status()
            java_result = java_response.json()  # 假设返回的是JSON
            print("Java API response:", java_result)
            # 提取JSON内容
            json_content = java_result.get("resultStr")

            # 参数 {"type":"easy_wav2lip","video_path":{"path":"ceyan.mp4","format":"mp4"},"audio_path":"baidu_9.wav","insert_index":0,"interrupt":true}
            json_msg = {
                "type": "easy_wav2lip",
                "video_path": {
                    "path":text,  # 默认视频路径
                    "format": "mp4"
                },
                "audio_path": tmpfile,
                "insert_index": 0,
                "interrupt": True
            }
            #将内容推送api接口到前端
            feature_recorded = cache.get(REDIS_WAITING_KEYS)
            print(f'feature_recorded: {feature_recorded}')
            # 判断feature_recorded是否存在
            if feature_recorded is not None:
                python_response = requests.post(API_MESSAGE, json=json_msg)
        return True

    def run(self):
        try:
            self.start_recording_and_vad()  # 监听语音流
            while not self.stop_event.is_set():
                self._duplex()  # 双工处理
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        finally:
            self.shutdown()



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="兴元机器人")

    # Add arguments
    parser.add_argument('--config_path', type=str, help="配置文件", default="../config/config.yaml")
    # Parse arguments
    args = parser.parse_args()
    config_path = args.config_path

    # 创建 Robot 实例并运行
    robot = Robot(config_path)
    robot.run()
