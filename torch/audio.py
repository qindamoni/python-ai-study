#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import utils
import librosa
import os

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

cnhubert_base_path = './models/'

class CNHubert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HubertModel.from_pretrained(cnhubert_base_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cnhubert_base_path
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats

def get_model():
    model = CNHubert()
    model.eval()
    return model

def get_tts(ref_wav_path,ref_text, ref_language, prompt_text, prompt_language):
    ref_text = ref_text.strip("\n")
    prompt_text = prompt_text.strip("\n")

    print('ref_text:',ref_text)
    print('prompt_text:',prompt_text)

    sovits_path = './s2G488k.pth'
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]

    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float32)
    print('zero_wav.shape:',zero_wav.shape)

    wav16k, sr = librosa.load(ref_wav_path, sr=16000)
    wav16k = torch.from_numpy(wav16k)
    print('wav16k.size',wav16k.size())
    zero_wav_torch = torch.from_numpy(zero_wav)
    print('zero_wav_torch.size',zero_wav_torch.size())

    wav16k = torch.cat([wav16k, zero_wav_torch])
    print('torch.cat',wav16k.size())

    ssl_model = get_model()

    ssl_content = ssl_model.model(wav16k.unsqueeze(0))
    exit()
    ssl_content = ssl_content["last_hidden_state"].transpose(1, 2)  # .float()
    codes = vq_model.extract_latent(ssl_content)
    prompt_semantic = codes[0, 0]

if __name__ == '__main__':
    ref_wav_path = './sun.wav'
    ref_text = '我的家在东北'
    ref_language = 'zh'

    prompt_text = '松花江上游'
    prompt_language = 'zh'

    get_tts(ref_wav_path,ref_text,ref_language,prompt_text,prompt_language)


'''
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (is_half == True):
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language.lower()]
    text_language = dict_language[text_language.lower()]
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)
    texts = text.split("\n")
    audio_bytes = BytesIO()

    for text in texts:
        # 简单防止纯符号引发参考音频泄露
        if only_punc(text):
            continue

        audio_opt = []
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config['inference']['top_k'],
                early_stop_num=hz * max_sec)
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if (is_half == True):
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                            refer).detach().cpu().numpy()[
                0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        audio_bytes = pack_audio(audio_bytes,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16),hps.data.sampling_rate)
    # logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        if stream_mode == "normal":
            audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
            yield audio_chunk
    
    if not stream_mode == "normal": 
        if media_type == "wav":
            audio_bytes = pack_wav(audio_bytes,hps.data.sampling_rate)
        yield audio_bytes.getvalue()
'''


