# URV5 Sequence Lite
 A lightweight URV5 batch sequence implementation tool

##What is this?
This is a lite URV5 implementation to separate sound files with multisteps. It try to makes separated sound files clearer.

Much more smaller than URV5 if you have pytorch environment and set include-system-site-packages = true.

Most code is borrowed from https://github.com/RVC-Boss/GPT-SoVITS

Following models are tested, which are enough for me. MDXnet is not implemented yet.
model_names = ['1_HP-UVR.pth',
               '5_HP-Karaoke-UVR.pth',
               '6_HP-Karaoke-UVR.pth',
               'bs_roformer_ep_317_sdr_12.9755.ckpt',
               'UVR-De-Echo-Aggressive.pth',
               'UVR-De-Echo-Normal.pth',
               'UVR-DeEcho-DeReverb.pth',
               'UVR-DeNoise-Lite.pth',
               'UVR-DeNoise.pth']


##How to use?
Taking separating music to clean voice as an example, the following 4 step is recommand:
BS-Roformer-Viperx-1297
5_HP-Karaoke-UVR
UVR-De-Echo
DeNoise(not necessary but just as an example)

You can use the following way to run.
```
model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', 'vocals', 'instrumental', None),
                    ('5_HP-Karaoke-UVR.pth', 'Karaoke_bg', 'Karaoke_main', 'vocals'),
                    ('UVR-De-Echo-Normal.pth', 'DeEcho_main', 'DeEcho_bg', 'Karaoke_main'),
                    ('UVR-DeNoise.pth', 'DeNoise_bg', 'DeNoise_main', 'DeEcho_main')]                  
                
filelist = [f'../resource/{i}.mp3' for i in ('music1', 'music')]
pipe = PipeLine(model_sequence, filelist, 'output')
pipe.run()
pipe.all2mp3() #Not recommand to call this function if ouput sounds are used to train model.
```
Every line in the model_sequence means the model name, the first separated file suffix, the second separated file suffix, and the input sound file suffix from last step.

##Why to write this?

It is ugly to use UVR5 with multi steps, like the output file will be like this:
-a----         2024/6/15     13:43       42009644 1_1_1_魔鬼中的天使_(Vocals)_(Vocals)_(Echo).wav
-a----         2024/6/15     13:43       42009644 1_1_1_魔鬼中的天使_(Vocals)_(Vocals)_(No Echo).wav
-a----         2024/6/15     13:42       42009644 1_1_魔鬼中的天使_(Vocals)_(Instrumental).wav
-a----         2024/6/15     13:42       42009644 1_1_魔鬼中的天使_(Vocals)_(Vocals).wav
-a----         2024/6/15     13:40       42011180 1_魔鬼中的天使_(Instrumental).wav
-a----         2024/6/15     13:40       42011180 1_魔鬼中的天使_(Vocals).wav
-a----         2024/6/15     13:43       46840364 2_2_2_小幸运_(Vocals)_(Vocals)_(Echo).wav
-a----         2024/6/15     13:43       46840364 2_2_2_小幸运_(Vocals)_(Vocals)_(No Echo).wav
-a----         2024/6/15     17:12       23420208 2_2_2_小幸运_Vocals_Vocals_NoEcho_FishLeong_0key_sovdiff_rmvpe.wav
-a----         2024/6/15     13:42       46840364 2_2_小幸运_(Vocals)_(Instrumental).wav
-a----         2024/6/15     13:42       46840364 2_2_小幸运_(Vocals)_(Vocals).wav
-a----         2024/6/15     13:41       46840552 2_小幸运_(Instrumental).wav
-a----         2024/6/15     13:41       46840552 2_小幸运_(Vocals).wav
-a----         2021/6/26     11:50        4249936 小幸运.mp3
-a----         2012/2/27     14:18        3856811 魔鬼中的天使.mp3

Many 1_1_1 or 2_2_2 and () are displayed. And, it does not support batch operations: multi files and multi steps.

##What skill shoud I have to use it?
Simple Python skills.
Senior Python environment install skills.


