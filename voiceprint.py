import os

import pyaudio
import wave
import librosa
import numpy as np
from pydub import AudioSegment  # 可选，仅用于后备方案



# 录制麦克风音频
def record_audio(seconds):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open('recorded_audio.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


# 读取本地音频文件
def read_audio_file(file_path):
    try:
        # 优先使用 soundfile（librosa 默认首选）
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        # 如果失败，尝试用 pydub 作为后备方案
        try:
            from pydub import AudioSegment
            print("====使用pydub读取音频文件===",file_path)
            boo = os.path.exists()
            print( "====音频文件是否存在====",boo)
            audio = AudioSegment.from_file(file_path)
            y = np.array(audio.get_array_of_samples())
            sr = audio.frame_rate
            # 如果是多声道，取单声道（librosa 默认单声道）
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            # 归一化到 [-1, 1]（librosa 默认范围）
            y = y / np.max(np.abs(y))
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
    return y, sr


# 对比两个音频文件的特征
def compare_audio_files():
    recorded_audio, sr_recorded = read_audio_file(r'D:\水务\水务\uploads\recording.wav')
    local_audio, sr_local = read_audio_file(r'D:\水务\水务\uploads\bert_vits2_29.wav')

    # 提取MFCC特征
    feature_recorded = np.mean(librosa.feature.mfcc(y=recorded_audio, sr=sr_recorded, n_mfcc=13), axis=1)
    feature_local = np.mean(librosa.feature.mfcc(y=local_audio, sr=sr_local, n_mfcc=13), axis=1)

    # 计算余弦相似度：
    similarity = np.dot(feature_recorded, feature_local) / (
                np.linalg.norm(feature_recorded) * np.linalg.norm(feature_local))
    print("音频1和音频2之间的相似度：", similarity)
    if similarity > 0.97:
        return True
    else:
        return False

# 调用函数获取音频特征
def get_audio_features(file_path):
    audio, sr = read_audio_file(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

#调用函数对比音频特征
def compare_audio_features(feature_recorded, feature_local):
    if feature_recorded is None:
        return False
    # 计算余弦相似度
    print("比对相似度feature_recorded, feature_local",feature_recorded, feature_local)
    similarity = np.dot(feature_recorded, feature_local) / (
            np.linalg.norm(feature_recorded) * np.linalg.norm(feature_local))
    print("音频1和音频2之间的相似度：", similarity)
    if similarity > 0.97:
        return True
    else:
        return False

# 调用函数录制麦克风音频
# record_audio(5)

# 对比录制的麦克风音频和本地音频文件
# result = compare_audio_files()
# print(result)

