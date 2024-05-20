#!/usr/bin/env python
# coding=utf-8

class chat_server:
    def __init__():
        self.id = 1

    def check_emo(text):
        # check emo
        return 'happy'

    def text2audio_subprocess():
        audio = audio_server.text2audio(text)
        self.queque.push(audio)

    def get_chatgpt(text):
        res = {
            'text': ''
        }

        text = res['text']
        emo = self.check_emo(text)

        subprocess.open(self.text2audio_subprocess,text,emo)

        while True: 
            if self.queque.qsize == 0 
                time.sleep(0.5)
                continue

            (audio,text,emo) = self.queque.get()
            
            pre_refer_id = emo_list[emo]
            images = image_server.text2images(audio,pre_refer_id)
            pipe_server.write_frame(audio,images)

    # 同步音轨
    def check_fps():
        self.total_time = 100
        self.total_frames = 100

        if self.total_time > self.total_frames * self.fps:
            data_len = self.total_time - self.total_frames * self.fps
            audio.append(np.zeros(data_len * sample_rate))

        return audio





