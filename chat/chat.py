#!/usr/bin/env python
# coding=utf-8
actions = {
    'happy':[
        {
            'start_idx': 4
            'audio': './actions/happy/act_0.wav'
            'images': './actions/happy/act_0'
        },
        {
            'start_idx': 15
            'audio': './actions/happy/act_1.wav'
            'images': './actions/happy/act_1'
        },
    ],
    'angry':[
        {
            'start_idx': 40
            'audio': './actions/angry/act_0.wav'
            'images': './actions/angry/act_0'
        },
        {
            'start_idx': 100
            'audio': './actions/angry/act_1.wav'
            'images': './actions/angry/act_1'
        },
    ]
}

def get_act(idx,imoj):
    if imoj == '':
        acts = acts

    curr_act = {}
    for i in acts:
       if i >= idx:
           curr_act = acts[i]

    return curr_act

def thread():
    msg_queque = queue.quere() 

    if msg_queque.qsize == 0:
        act = get_act(image_server.idx)

        pip_server.write_frame(act['audio'],act['images'])
    else:
        text = msg_queque.get()
        audio = audio_server.text2audio(text)
        images = image_server.audio2image(audio)

        pip_server.write_frame(audio,images)







