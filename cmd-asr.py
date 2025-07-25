# -*- coding:utf-8 -*-

from flask import Flask, render_template, request, jsonify
import sys,os,traceback
import torch
from funasr import AutoModel

from redisManage import VoiceFeatureCache
from thirdCompare import RealTimeVoiceVerifier
from voiceprint import get_audio_features,compare_audio_features
from flask_caching import Cache
from flask_cors import CORS

import time
app = Flask(__name__)

app.config["CACHE_TYPE"] = "RedisCache"
app.config["CACHE_REDIS_HOST"] = "127.0.0.1"  # Redis 服务器地址
app.config["CACHE_REDIS_PORT"] = 6379       # Redis 端口
app.config["CACHE_REDIS_DB"] = 1      # 使用的数据库编号
app.config["CACHE_DEFAULT_TIMEOUT"] = 300    # 默认缓存5分钟
cache = Cache(app)

#批量添加跨域
CORS(app, resources={
    r"/vop": {"origins": "*"},
    r"/vop/v2": {"origins": "*"},
    r"/delcache": {"origins": "*"},
    r"/addcache": {"origins": "*"},
})
# CORS(app, resources={r"/vop/v2": {"origins": "*"}})
# CORS(app, resources={r"/delcache": {"origins": "*"}})
# CORS(app, resources={r"/delcache/v2": {"origins": "*"}})
# CORS(app, resources={r"/addcache": {"origins": "*"}})
# CORS(app, resources={r"/vop": {"origins": "*"}})
# CORS(app, resources={r"/vop/v2": {"origins": "*"}})
#帮把上述代码写到一起


# dir=r"E:\project\shuiwu\shuiwu\uploads";
# if(dir[-1]=="/"):dir=dir[:-1]
# opt_name=dir.split("\\")[-1].split("/")[-1]

path_asr='./models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad='./models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc='./models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr=path_asr if os.path.exists(path_asr)else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad=path_vad if os.path.exists(path_vad)else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc=path_punc if os.path.exists(path_punc)else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

torch.cuda.set_per_process_memory_fraction(0.3)  # 限制为 50% 显存
torch.cuda.empty_cache()  # 清空缓存

# 多进程和 CUDA 设置
torch.multiprocessing.set_start_method('spawn', force=True)
torch.backends.cudnn.benchmark = True

model = AutoModel(model=path_asr, model_revision="v2.0.4",
                disable_update=True,
                  vad_model=path_vad,
                  vad_model_revision="v2.0.4",
                  punc_model=path_punc,
                  punc_model_revision="v2.0.4",
                  )

V_BODY="lambda_03"

@app.route("/vop", methods=["GET"])
def vop():
    file_path = request.args.get("file_path")  # 从 ?file_path=xxx 获取
    if not file_path:
        return jsonify({"error": "缺少 file_path 参数"}), 400
    client_ip = request.remote_addr
    if file_path[-1] == "/": file_path = file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    # 获取当前播报的声纹进行保存
    # body_features = get_audio_features("%s/%s" % ("D:\\shuiwu\\asrcache\\asr\\asr", V_BODY + ".wav"))
    # print("=========vbody声源存入成功=======",body_features)
    # 将body_features 放到缓存
    # cache.set(V_BODY,body_features)
    # 获取当前时间戳
    timestamp = int(time.time())
    # 生成key名
    key_name = f"{client_ip}"
    for name in file_names:
        try:
            # 获取本地音频特征
            feature_local = get_audio_features("%s/%s" % (file_path, name))
            # 获取音频特征
            # v_feature = cache.get(V_BODY)
            # print("缓存内声纹======",v_feature)
            # print("系统声纹=======",body_features)
            # print("本地声纹=======",feature_local)
            # if compare_audio_features(feature_local, body_features) is True:
            #     print("=========匹配的声纹是系统声音==============")
            #     text = ""
            #     opt[-1] = "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text)
            #     return jsonify(opt)
            # print("==========非系统声音的声纹================")
            text = model.generate(input="%s/%s" % (file_path, name))[0]["text"]
            opt.append("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            print("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            # 判断是否存在关键字 "小园", "小原", "晓园", "小员", "小圆", "小袁",
            #                     "小猿", "小缘", "小辕", "小媛", "小元", "小源",
            #                     "晓媛", "晓园"
            if any(keyword in text for keyword in ["你好小园", "你好小原", "你好小员", "你好小圆", "你好小袁",
                                                   "你好小猿", "你好小缘", "你好小辕", "你好小媛", "你好小元",
                                                   "你好小源",
                                                   "你好，小袁", "你好，小猿", "你好，小圆", "你好，小园",
                                                   "你好，小原", "你好，小元", "你好，小源", "你好，小辕"]):
                print("==============检测到关键字=================")
                # 进行声纹保存 基于 client_ip和当前时间戳 存入cache
                # # 获取音频特征
                audio_features = get_audio_features("%s/%s" % (file_path, name))
                cache.set(key_name, audio_features, timeout=60 * 60 * 24)  # 设置缓存过期时间为24小时
            # 对比声纹
            # 获取缓存中的音频特征
            feature_recorded = cache.get(key_name)
            print("================获取缓存内容==========", feature_recorded)
            if feature_recorded is None:
                text = ""
                opt[-1] = "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text)
                return jsonify(opt)
            # compare_audio_features 进行判断
            if compare_audio_features(feature_recorded, feature_local):
                print("声纹对比成功")
                # 进行后续操作,目前暂时不需要做处理,预留接口
            else:
                print("声纹对比失败")
                # 清空 opt中的text数据
                text = ""
            opt[-1] = "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text)
            # os.remove("%s/%s" % (file_path, name))
        except:
            print(traceback.format_exc())
    return jsonify(opt)

#删除缓存接口
@app.route("/delcache", methods=["GET"])
def delcache():
    client_ip = request.remote_addr
    cache.delete(client_ip)
    return jsonify({"message": "缓存删除成功"}), 200

@app.route("/vop/v2", methods=["GET"])
def vopV2():
    file_path = request.args.get("file_path")  # 从 ?file_path=xxx 获取
    if not file_path:
        return jsonify({"error": "缺少 file_path 参数"}), 400
    cache_names  = ['first voice', 'second voice', 'third voice', 'fourth voice', 'fifth voice']
    client_ip = request.remote_addr
    if file_path[-1]== "/":file_path= file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    # 生成key名
    key_name = f"{client_ip}"
    for name in file_names:
        for cache_name in cache_names:
            try:
                redis_cache_name = client_ip +'_'+ cache_name
                # 获取本地音频特征
                feature_local = get_audio_features("%s/%s" % (file_path, name))
                # 获取音频特征
                print("本地声纹=======",feature_local)
                #对比声纹
                # 获取缓存中的音频特征
                print(redis_cache_name)
                feature_recorded = cache.get(redis_cache_name)
                print("缓存内声纹======",feature_recorded)
                #compare_audio_features 进行判断
                if compare_audio_features(feature_recorded, feature_local):
                    print("声纹对比成功")
                    # 进行后续操作,目前暂时不需要做处理,预留接口
                    text = '匹配缓存是 '+cache_name
                    if opt == []:
                        opt.append("%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text,100))
                    else:
                        opt[-1] = "%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text,100)
                    return jsonify(opt)
                else:
                    print("声纹对比失败")
                    #清空 opt中的text数据
                    text = "声纹对比失败"
                    if opt == []:
                        opt.append("%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text,0))
                    else:
                        opt[-1] = "%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text,0)
                # os.remove("%s/%s" % (file_path, name))
            except:
                print(traceback.format_exc())
    return jsonify(opt)

# 初始化缓存和验证器
cache = VoiceFeatureCache(host='120.211.84.149', port=6379, db=0)
verifier = RealTimeVoiceVerifier(feature_cache=cache)

@app.route("/vop/v3", methods=["GET"])
def vopV3():
    file_path = request.args.get("file_path")  # 从 ?file_path=xxx 获取
    if not file_path:
        return jsonify({"error": "缺少 file_path 参数"}), 400
    client_ip = request.remote_addr
    if file_path[-1] == "/": file_path = file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    # 获取当前播报的声纹进行保存
    # body_features = get_audio_features("%s/%s" % ("D:\\shuiwu\\asrcache\\asr\\asr", V_BODY + ".wav"))
    # print("=========vbody声源存入成功=======",body_features)
    # 将body_features 放到缓存
    # cache.set(V_BODY,body_features)
    # 获取当前时间戳
    timestamp = int(time.time())
    # 生成key名
    key_name = f"{client_ip}"
    for name in file_names:
        try:
            # 获取本地音频特征
            feature_local = get_audio_features("%s/%s" % (file_path, name))
            # 获取音频特征
            # v_feature = cache.get(V_BODY)
            # print("缓存内声纹======",v_feature)
            # print("系统声纹=======",body_features)
            # print("本地声纹=======",feature_local)
            # if compare_audio_features(feature_local, body_features) is True:
            #     print("=========匹配的声纹是系统声音==============")
            #     text = ""
            #     opt[-1] = "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text)
            #     return jsonify(opt)
            # print("==========非系统声音的声纹================")
            text = model.generate(input="%s/%s" % (file_path, name))[0]["text"]
            opt.append("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            print("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            # 判断是否存在关键字 "小园", "小原", "晓园", "小员", "小圆", "小袁",
            #                     "小猿", "小缘", "小辕", "小媛", "小元", "小源",
            #                     "晓媛", "晓园"
            if any(keyword in text for keyword in ["你好小园", "你好小原", "你好小员", "你好小圆", "你好小袁",
                                                   "你好小猿", "你好小缘", "你好小辕", "你好小媛", "你好小元",
                                                   "你好小源",
                                                   "你好，小袁", "你好，小猿", "你好，小圆", "你好，小园",
                                                   "你好，小原", "你好，小元", "你好，小源", "你好，小辕"]):
                print("==============检测到关键字=================")
                # 进行声纹保存 基于 client_ip和当前时间戳 存入cache
                # # 获取音频特征
                # audio_features = get_audio_features("%s/%s" % (file_path, name))
                # 注册用户
                verifier.enroll_user(
                    user_id=client_ip,
                    audio_path=file_path+"/"+name,
                    ttl= 86400  # 保留1天
                )
                # cache.set(key_name, audio_features, timeout=60 * 60 * 24)  # 设置缓存过期时间为24小时
            # 对比声纹
            # 获取缓存中的音频特征
            # 验证流程
            result = verifier.verify(
                test_audio_path=file_path+"/"+name,
                user_id=client_ip
            )
            print(f"验证结果: {result['result']}")
            print(f"总耗时: {result['total_time'] * 1000:.2f}ms")
            if result['result']:
                print("声纹对比成功")
                # 进行后续操作,目前暂时不需要做处理,预留接口
            else:
                print("声纹对比失败")
                # 清空 opt中的text数据
                text = ""
            opt[-1] = "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text)
            # os.remove("%s/%s" % (file_path, name))
        except:
            print(traceback.format_exc())
    return jsonify(opt)

# 不加声纹识别的语音识别接口
@app.route("/vop/nosign", methods=["GET"])
def vop_nosign():
    file_path = request.args.get("file_path")  # 从 ?file_path=xxx 获取
    if not file_path:
        return jsonify({"error": "缺少 file_path 参数"}), 400
    client_ip = request.remote_addr
    if file_path[-1] == "/": file_path = file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    # 生成key名
    key_name = f"{client_ip}"
    for name in file_names:
        try:
            # 获取本地音频特征
            text = model.generate(input="%s/%s" % (file_path, name))[0]["text"]
            opt.append("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            #os.remove("%s/%s" % (file_path, name))
        except:
            print(traceback.format_exc())
    return jsonify(opt)

# 删除缓存接口
#允许跨域
@app.route("/delcache/v2", methods=["GET"])
def delcacheV2():
    cache_name = request.args.get("cache_name")
    if not cache_name:
        return jsonify({"error": "缺少 cache_name 参数"}), 400
    cache.delete(cache_name)
    return jsonify({"message": "缓存删除成功"}), 200
#新增缓存接口
@app.route("/addcache", methods=["GET"])
def addcache():
    print( "添加缓存接口")
    file_path = request.args.get("file_path")
    if not file_path:
        return jsonify({"error": "缺少 file_path 参数"}), 400
    cache_name = request.args.get("cache_name")
    if not cache_name:
        return jsonify({"error": "缺少 cache_name 参数"}), 400
    client_ip = request.remote_addr
    if file_path[-1] == "/": file_path = file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    # 生成key名
    key_name = f"{client_ip}"
    for name in file_names:
        # 获取音频特征
        audio_features = get_audio_features("%s/%s" % (file_path, name))
        # 将音频特征存入缓存
        cache.set(client_ip + "_"+cache_name, audio_features, timeout=60 * 60 * 24)
    return jsonify({"message": "缓存添加成功"}), 200

import librosa
import soundfile as sf
import numpy as np
CHUNK_SIZE =640
ENCODER_CHUNK_LOOK_BACK = 6
DECODER_CHUNK_LOOK_BACK = 4
# 获取当前脚本的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
@app.route("/audio", methods=["GET"])
def check_local_file():
    # 加载并预处理音频
    audio_path = "test.wav"
    file = os.path.join(script_dir, 'static', 'test.wav')
    audio_data, sr = sf.read(file)
    # 重采样至 16kHz
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    # 转为单声道
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 归一化并转为 int16
    audio_data = librosa.util.normalize(audio_data)
    audio_array = (audio_data * 32767).astype(np.int16)
    # 初始化缓存
    cache = None
    results = []
    text_result = []
    # 模拟按 chunk 输入
    for start in range(0, len(audio_array), CHUNK_SIZE):
        chunk = audio_array[start: start + CHUNK_SIZE]
        is_final = start + CHUNK_SIZE >= len(audio_array)  # 最后一块标记 is_final
        # 带异常捕获的生成
        try:
            resultT = model.generate(
                input=chunk[np.newaxis, :],  # 添加批次维度
                cache={},
                is_final=is_final,
                chunk_size=CHUNK_SIZE,
                encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK
            )
            if isinstance(resultT, list):
                # 提取列表中第一个元素的文本（典型结构：多个结果按置信度排序）
                text = resultT[0].get("text", "") if resultT else ""
            elif isinstance(resultT, dict):
                text = resultT.get("text", "")
            else:
                text = ""
            results.append(text)
        except Exception as e:
            print(f"生成过程异常: {str(e)}")

    # 最后一帧标记结束
    final_result = model.generate(
        input=np.zeros((1, 0), dtype=np.int16),  # 空输入用于标记 is_final
        cache=cache,
        is_final=True,
        chunk_size=CHUNK_SIZE,
        encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
        decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
    )

    final_text = ""
    if isinstance(final_result, dict):
        final_text = final_result.get("text", "")
    elif isinstance(final_result, list):
        final_text = final_result[0].get("text", "")

    print("\n🧾 最终识别结果：")
    print("".join(results) + final_text)


if __name__ == '__main__':
    app.run(debug=True, port=5003)