#!/usr/bin/env python
# coding=utf-8
import wave
from sys import getsizeof
import numpy as np
import ffmpeg
import scipy.signal as sps
from scipy.io import wavfile
from io import BytesIO
import librosa
import subprocess

'''
f = wave.open(r"sun.wav","rb")
params = f.getparams()

nchannels, sampwidth, framerate, nframes = params[:4]
print('params:',params)

frames = f.readframes(2)
print(frames)
print('frames len:',len(frames))
print('frames 0 tyep:',type(frames[0]))
print('frames 0 :',frames[0])

print('=========')

np_int16 = np.frombuffer(frames,dtype=np.int16)
print(np_int16)
print(np_int16 / 2)
print('np_int16 len:',len(np_int16))
print('np_init16 0 type:',type(np_int16[0]))
print('np_init16 0 :',np_int16[0])

print('=========')

np_float32 = np_int16.astype(np.float32)
print(np_float32)
print('np_float32 len:',len(np_float32))
print('np_float32 0 type:',type(np_float32[0]))
print('np_float32 0 :',np_float32[0])

audio_bytes = BytesIO()

concat = np.concatenate([np_int16,np_int16],0)
print(concat)
concat = concat * 32768
print(concat)
concat = concat.astype(np.int16)
print(concat)
concat = concat.tobytes()
audio_bytes.write(concat)
print(concat)
print(audio_bytes)
concat = audio_bytes.getvalue()
print(concat)

np_float32 = np.concatenate(audio_opt, 0)
audio_bytes = pack_audio(audio_bytes,
                         (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
                         hps.data.sampling_rate)
    audio_bytes.write(data.tobytes())

audio_bytes = pack_wav(audio_bytes,hps.data.sampling_rate)
    data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
    wav_bytes = BytesIO()
    sf.write(wav_bytes, data, rate, format='wav')
    return wav_bytes

yield audio_bytes.getvalue()

text2audio(text)
output:float32,sampling_rate=32000

audio2image(audio)
input:int16,sampling_rate=16000
'''

def text2audio_load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le",ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

def audio2image_load_audio(file,sr):
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    #return np.frombuffer(out, np.int16)

def resample(data,origin,target):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 'f32le',  # 输入16位有符号小端整数PCM
        '-ar', f'{origin}',  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-ar', f'{target}',
        '-vn',  # 不包含视频
        '-f', 'f32le',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    return np.frombuffer(out, np.float32).flatten()

'''
data = text2audio_load_audio('./sun.wav',44100)
print('========text2audio_load_audio:f32le,32000')
print(data[:4])

print('=========resample to 16000')
sample_rate, clip = wavfile.read('./sun.wav')
print(sample_rate)
print('clip len:',len(clip))
print('clip type:',type(clip[0]))
print('clip :',clip[:4])
#print(len(data))
#print(type(data[0]))
clip=clip.flatten().astype(np.float32) / 32768
print('===========clip to float32')
print('clip len:',len(clip))
print('clip type:',type(clip[0]))
print('clip :',clip[:4])

number_of_samples = round(len(clip) * float(16000) / 44100)
clip = sps.resample(clip, number_of_samples)
print('===========clip to sample 16000')
print('number of samples:',number_of_samples)
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:4])
#clip = clip.astype(np.int16)
'''

#clip,sample = librosa.load('./sun.wav',sr=16000,dtype=np.float32)
clip = text2audio_load_audio('./sun.wav',32000)
print('=====text2audio_load_audio')
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:8])

'''
clip = text2audio_load_audio('./sun.wav',32000)
print('=====librosa.load sample:')
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:8])
'''

clip = resample(clip,32000,16000)
print('======resample 32000 to 16000')
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:8])

clip = text2audio_load_audio('./sun.wav',32000)
print('=====sample to 16000')
clip = librosa.resample(clip,orig_sr=32000,target_sr=16000)
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:8])

sample_rate, clip = wavfile.read('./sun.wav')
number_of_samples = round(len(clip) * float(16000) / sample_rate)
clip = sps.resample(clip, number_of_samples).astype(np.float32) / 32768
print('=====sample to 16000')
print('clip type:',type(clip[0]))
print('clip len:',len(clip))
print('cilp :',clip[:8])

data = audio2image_load_audio('./sun.wav',16000)
print('======text2audio_load_audio:s16le,16000')
print('data type:',type(data[0]))
print('data len:',len(data))
print('data :',data[:8])

'''
from scipy.io import wavfile
import scipy.signal as sps
from io import BytesIO

new_rate = 2000
# Read file
sample_rate, clip = wavfile.read(BytesIO(file_name))

# Resample data
number_of_samples = round(len(clip) * float(new_rate) / sample_rate)
clip = sps.resample(clip, number_of_samples)

f = wave.open(r"sun.wav","rb")
frames = f.readframes(4)
print(frames)
np_int16 = np.frombuffer(frames,dtype=np.int16)
print(np_int16)
np_float32 = np_int16.flatten().astype(np.float32) / 32768.0
print(np_float32)
np_int16 =  (np_float32 * 32768).astype(np.int16)
print(np_int16)
'''
