import redis
import msgpack
import base64
import numpy as np

from functools import lru_cache
class VoiceFeatureCache:
    local_cache_size = 100
    """声纹特征缓存管理器（Redis + 本地LRU缓存）"""

    def __init__(self, host='120.211.84.149', port=6379, db=1, local_cache_size=100):
        self.redis_pool = redis.ConnectionPool(host=host, port=port, db=db)
        self.local_cache_size = local_cache_size

    def _serialize_feature(self, feature):
        """序列化特征字典"""
        packed = {
            'mfcc': feature['mfcc'].astype(np.float32).tobytes(),
            'spectral_centroid': feature['spectral_centroid'].astype(np.float32).tobytes(),
            'zcr': feature['zcr'].astype(np.float32).tobytes(),
            'shapes': {
                'mfcc': feature['mfcc'].shape,
                'spectral_centroid': feature['spectral_centroid'].shape,
                'zcr': feature['zcr'].shape
            }
        }
        # 序列化时（_serialize_feature 函数内）
        print("[Serialization] spectral_centroid shape:", feature['spectral_centroid'].shape)
        print("[Serialization] spectral_centroid dtype:", feature['spectral_centroid'].dtype)
        print("[Serialization] spectral_centroid bytes length:", len(packed['spectral_centroid']))
        return base64.b64encode(msgpack.packb(packed))

    def _deserialize_feature(self, data):
        """反序列化特征字典"""
        unpacked = msgpack.unpackb(base64.b64decode(data))
        # 反序列化时（_deserialize_feature 函数内）
        spectral_bytes = unpacked['spectral_centroid']
        print("[Deserialization] spectral_centroid bytes length:", len(spectral_bytes))
        spectral_data = np.frombuffer(spectral_bytes, dtype=np.float32)
        print("[Deserialization] spectral_data raw shape:", spectral_data.shape)
        return {
            'mfcc': np.frombuffer(unpacked['mfcc'], dtype=np.float32).reshape(unpacked['shapes']['mfcc']),
            'spectral_centroid': np.frombuffer(unpacked['spectral_centroid'], dtype=np.float32).reshape(
                unpacked['shapes']['spectral_centroid']),
            'zcr': np.frombuffer(unpacked['zcr'], dtype=np.float32).reshape(unpacked['shapes']['zcr'])
        }

    @lru_cache(maxsize=local_cache_size)
    def get_feature(self, user_id):
        """获取特征（带本地缓存）"""
        with redis.Redis(connection_pool=self.redis_pool) as conn:
            data = conn.get(f"voiceprint:{user_id}")
            return self._deserialize_feature(data) if data else None

    def save_feature(self, user_id, feature, ttl=86400):
        """保存特征到Redis并更新本地缓存"""
        with redis.Redis(connection_pool=self.redis_pool) as conn:
            conn.setex(
                name=f"voiceprint:{user_id}",
                value=self._serialize_feature(feature),
                time=ttl
            )
        # 清除旧缓存
        self.get_feature.cache_clear()

    def delete(self ,user_id):
        """删除特征"""
        with redis.Redis(connection_pool=self.redis_pool) as conn:
            conn.delete(f"voiceprint:{user_id}")
        # 清除旧缓存
        self.get_feature.cache_clear()
