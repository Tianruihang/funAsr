import os
import time
import traceback

from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_cors import CORS

from redisManage import VoiceFeatureCache
from thirdCompare import RealTimeVoiceVerifier
from voiceprint import get_audio_features

app = Flask(__name__)

app.config["CACHE_TYPE"] = "RedisCache"
app.config["CACHE_REDIS_HOST"] = "120.211.84.149"  # Redis 服务器地址
app.config["CACHE_REDIS_PORT"] = 6379       # Redis 端口
app.config["CACHE_REDIS_DB"] = 1      # 使用的数据库编号
app.config["CACHE_DEFAULT_TIMEOUT"] = 300    # 默认缓存5分钟


#批量添加跨域
CORS(app, resources={
    r"/vop": {"origins": "*"},
    r"/vop/v2": {"origins": "*"},
    r"/delcache": {"origins": "*"},
    r"/addcache": {"origins": "*"},
})


V_BODY="lambda_03"

# 初始化缓存和验证器
cache = VoiceFeatureCache(host='120.211.84.149', port=6379, db=1)
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
            # feature_local = get_audio_features("%s/%s" % (file_path, name))
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

            text = ""
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
        # 注册用户
        print("注册用户",file_path + "/" + name)
        verifier.enroll_user(
            user_id=cache_name,
            audio_path=file_path + "/" + name,
            ttl=86400  # 保留1天
        )
    return jsonify({"message": "缓存添加成功"}), 200



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
    resultBoo = False
    for name in file_names:
        for cache_name in cache_names:
            try:
                # 获取音频特征
                #对比声纹
                # 获取缓存中的音频特征
                #compare_audio_features 进行判断
                # 验证流程
                if(cache.get_feature(cache_name) is None):
                    print("缓存不存在")
                    continue
                result = verifier.verify(
                    test_audio_path=file_path + "/" + name,
                    user_id=cache_name
                )
                if result["result"]:
                    resultBoo = True
                    text = '匹配缓存是 ' + cache_name
                print("验证文件",file_path + "/" + name)
                print(f"cache_name: {cache_name}")

                # os.remove("%s/%s" % (file_path, name))
            except:
                print(traceback.format_exc())
    if resultBoo:
        print("声纹对比成功")
        # 进行后续操作,目前暂时不需要做处理,预留接口
        # text = '匹配缓存是 ' + cache_name
        if opt == []:
            opt.append("%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text, 100))
        else:
            opt[-1] = "%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text, 100)
        return jsonify(opt)
    else:
        print("声纹对比失败")
        # 清空 opt中的text数据
        text = "声纹对比失败"
        if opt == []:
            opt.append("%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text, 0))
        else:
            opt[-1] = "%s/%s|%s|ZH|%s|%s" % (file_path, name, opt_name, text, 0)
    return jsonify(opt)


@app.route("/compare/video", methods=["GET"])
def compare():
    # 获取音频特征
    # 获取当前文件目录
    file_path = "E:\\project\\PRO_FACERECO\\uploadCacheTest\\qiepian"
    name1 = request.args.get("name")
    # 将音频特征存入缓存
    verifier.enroll_user(
        user_id="test_user",
        audio_path=file_path + "/" + name1,
        ttl=86400  # 保留1天
    )
    file_names = os.listdir(file_path)
    file_names.sort()
    for name in file_names:
        try:
            result = verifier.verify(
                test_audio_path=file_path + "/" + name,
                user_id="test_user"
            )
            if result['result']:
                print("声纹对比成功",result)
                # 进行后续操作,目前暂时不需要做处理,预留接口
            else:
                print("声纹对比失败")
        except:
            print(traceback.format_exc())
    return jsonify("ok")

if __name__ == '__main__':
    app.run(debug=True, port=5003)