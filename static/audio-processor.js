// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];

    // 1. 重采样逻辑（48kHz → 16kHz）
    if (input.length > 0) {
      const resampled = this.resample(input[0], 48000, 16000); // 输入采样率需动态检测
      // 2. 将处理后的数据复制到输出
      for (let i = 0; i < Math.min(resampled.length, output[0].length); i++) {
        output[0][i] = resampled[i];
      }
    }
    return true;
  }

  // 重采样方法（整数倍降采样）
  resample(audioData, sourceRate, targetRate) {
    const ratio = sourceRate / targetRate;
    const result = new Float32Array(Math.floor(audioData.length / ratio));
    for (let i = 0; i < result.length; i++) {
      let sum = 0;
      for (let j = 0; j < ratio; j++) {
        sum += audioData[i * ratio + j];
      }
      result[i] = sum / ratio; // 移动平均降采样
    }
    return result;
  }
}
registerProcessor('audio-processor', AudioProcessor);