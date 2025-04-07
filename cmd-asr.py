# -*- coding:utf-8 -*-
from flask import Flask, render_template, request, jsonify
import sys,os,traceback
import torch
from funasr import AutoModel

app = Flask(__name__)

# dir=r"E:\project\shuiwu\shuiwu\uploads";
# if(dir[-1]=="/"):dir=dir[:-1]
# opt_name=dir.split("\\")[-1].split("/")[-1]

path_asr='/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad='/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc='/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr=path_asr if os.path.exists(path_asr)else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad=path_vad if os.path.exists(path_vad)else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc=path_punc if os.path.exists(path_punc)else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

torch.cuda.set_per_process_memory_fraction(0.5)  # 限制为 50% 显存
torch.cuda.empty_cache()  # 清空缓存

# 多进程和 CUDA 设置
torch.multiprocessing.set_start_method('spawn', force=True)
torch.backends.cudnn.benchmark = True

model = AutoModel(model=path_asr, model_revision="v2.0.4",
                  vad_model=path_vad,
                  vad_model_revision="v2.0.4",
                  punc_model=path_punc,
                  punc_model_revision="v2.0.4",
                  )


@app.route("/vop", methods=["GET"])
def vop():
    file_path = request.args.get("file_path")  # 从 ?file_path=xxx 获取
    if file_path[-1]== "/":file_path= file_path[:-1]
    opt_name = os.path.basename(file_path)
    # 调用接口识别语音结果
    opt = []
    file_names = os.listdir(file_path)
    file_names.sort()
    for name in file_names:
        try:
            text = model.generate(input="%s/%s" % (file_path, name))[0]["text"]
            opt.append("%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            print( "%s/%s|%s|ZH|%s" % (file_path, name, opt_name, text))
            os.remove("%s/%s" % (file_path, name))
        except:
            print(traceback.format_exc())
    return jsonify(opt)

if __name__ == '__main__':
    app.run(debug=True, port=5003)