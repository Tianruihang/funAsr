import librosa
import numpy as np
import time
import redis
import msgpack
import base64
from functools import lru_cache
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw

from redisManage import VoiceFeatureCache


class RealTimeVoiceVerifier:
    """支持多模板的实时声纹验证器"""

    def __init__(self, feature_cache: VoiceFeatureCache, thresholds=None):
        """
        参数:
            feature_cache: 特征缓存实例
            thresholds: 验证阈值配置
        """
        self.cache = feature_cache
        self.thresholds = thresholds or {
            'zcr': 0.0559,
            'sc': 0.0045,
            'mfcc': 185.0
        }
        self.candidate_count = 0
        self.full_compare_count = 0

    @staticmethod
    def extract_features(audio_path):
        """通用特征提取方法"""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # 标准化音频长度
        target_length = 22050 * 3  # 3秒
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # 并行特征计算
        return {
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512).T,
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).T,
            'zcr': librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512).T
        }

    def _zcr_compare(self, enroll_feat, test_feat):
        """零交叉率快速比对"""
        enroll_zcr = np.mean(enroll_feat['zcr'], axis=0)
        test_zcr = np.mean(test_feat['zcr'], axis=0)
        return euclidean(enroll_zcr, test_zcr)

    def _sc_compare(self, enroll_feat, test_feat):
        """谱质心比对"""
        enroll_sc = np.hstack([
            np.mean(enroll_feat['spectral_centroid']),
            np.std(enroll_feat['spectral_centroid'])
        ])
        test_sc = np.hstack([
            np.mean(test_feat['spectral_centroid']),
            np.std(test_feat['spectral_centroid'])
        ])
        return cosine(enroll_sc, test_sc)

    def _mfcc_compare(self, enroll_feat, test_feat):
        """动态时间规整比对"""
        distance, _ = fastdtw(
            enroll_feat['mfcc'],
            test_feat['mfcc'],
            dist=euclidean
        )
        return distance / len(enroll_feat['mfcc'])

    def verify(self, test_audio_path, user_id):
        """
        执行声纹验证流程
        返回:
            dict: 包含验证结果、阶段耗时等信息
        """
        self.candidate_count += 1
        start_time = time.time()

        # 获取注册模板
        enroll_feat = self.cache.get_feature(user_id)
        if enroll_feat is None:
            raise ValueError(f"用户 {user_id} 的声纹模板不存在")
        # 提取测试音频特征
        test_feat = self.extract_features(test_audio_path)

        # 第一阶段：零交叉率筛查
        stage1_start = time.time()
        zcr_dist = self._zcr_compare(enroll_feat, test_feat)
        sc_dist = self._sc_compare(enroll_feat, test_feat)
        mfcc_dist = self._mfcc_compare(enroll_feat, test_feat)
        # self.thresholds['zcr'] * 0.2 + self.thresholds['sc'] * 0.4 + self.thresholds['mfcc'] * 0.4
        defaultRate =  self.thresholds['zcr'] * 357.78 + self.thresholds['sc'] * 3750 + self.thresholds['mfcc'] * 0.3514
        print("defaultRate",defaultRate)
        # zcr_dist * 20% + sc_dist * 40% + mfcc_dist * 40%
        compareRate = zcr_dist * 357.78 + sc_dist * 3750 + mfcc_dist *  0.3514
        print("compareRate",compareRate)
        if compareRate > defaultRate:
            print(f'验证失败: zcr_dist={zcr_dist}, sc_dist={sc_dist}, mfcc_dist={mfcc_dist}')
            return {
                'result': False,
                'stage': 0,
                'total_time': time.time() - start_time,
                'zcr_dist': zcr_dist,
                'sc_dist': sc_dist,
                'mfcc_dist': mfcc_dist
            }
        else:
            print(f'验证成功: zcr_dist={zcr_dist}, sc_dist={sc_dist}, mfcc_dist={mfcc_dist}')
            return {
                'result': True,
                'stage': 0,
                'total_time': time.time() - start_time,
                'zcr_dist': zcr_dist,
                'sc_dist': sc_dist,
                'mfcc_dist': mfcc_dist
            }
        # 计算加权距离

        # print("zcr_dist", zcr_dist)
        # if zcr_dist > self.thresholds['zcr']:
        #     return {
        #         'result': False,
        #         'stage': 1,
        #         'total_time': time.time() - start_time,
        #         'zcr_dist': zcr_dist
        #     }
        #
        # # 第二阶段：谱质心比对
        # stage2_start = time.time()
        #
        # print("sc_dist", sc_dist)
        # if sc_dist > self.thresholds['sc']:
        #     return {
        #         'result': False,
        #         'stage': 2,
        #         'total_time': time.time() - start_time,
        #         'zcr_dist': zcr_dist,
        #         'sc_dist': sc_dist
        #     }
        #
        # # 第三阶段：MFCC精细比对
        # self.full_compare_count += 1
        # total_time = time.time() - start_time
        # print("mfcc_dist",mfcc_dist)
        # return {
        #     'result': mfcc_dist < self.thresholds['mfcc'],
        #     'stage': 3,
        #     'total_time': total_time,
        #     'zcr_dist': zcr_dist,
        #     'sc_dist': sc_dist,
        #     'mfcc_dist': mfcc_dist
        # }

    def enroll_user(self, user_id, audio_path, ttl=86400):
        """注册用户声纹模板"""
        features = self.extract_features(audio_path)
        self.cache.save_feature(user_id, features, ttl)
        return True





