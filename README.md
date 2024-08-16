# URV5 Sequence Lite
 A lightweight URV5 batch sequence implementation tool.

## What is this?
This is a lite URV5 implementation to separate sound files with multisteps. It try to makes separated sound files clearer.

Much more smaller than URV5 if you have pytorch environment when setting include-system-site-packages = true.

Most code is borrowed from https://github.com/RVC-Boss/GPT-SoVITS

Following models are tested, which are enough for me. MDXnet is not implemented yet.
```
model_names = ['1_HP-UVR.pth',
               '5_HP-Karaoke-UVR.pth',
               '6_HP-Karaoke-UVR.pth',
               'bs_roformer_ep_317_sdr_12.9755.ckpt',
               'UVR-De-Echo-Aggressive.pth',
               'UVR-De-Echo-Normal.pth',
               'UVR-DeEcho-DeReverb.pth',
               'UVR-DeNoise-Lite.pth',
               'UVR-DeNoise.pth']
```

## How to use?
Taking separating music to clean voice as an example, the following 4 step is recommanded:
1. BS-Roformer-Viperx-1297
2. 5_HP-Karaoke-UVR
3. UVR-De-Echo
4. DeNoise (not necessary but just as an example)

You can use the following way to run.
```Python3
import UVR5_lite
def main_pipe():
    model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', 'vocals', 'instrumental', None),
                     ('5_HP-Karaoke-UVR.pth', 'Karaoke_bg', 'Karaoke_main', 'vocals'),
                     ('UVR-De-Echo-Normal.pth', 'DeEcho_main', 'DeEcho_bg', 'Karaoke_main'),
                     ('UVR-DeNoise.pth', 'DeNoise_bg', 'DeNoise_main', 'DeEcho_main')]                  
                  
    filelist = [f'../resource/{i}.mp3' for i in ('music1', 'music2')]
    pipe = UVR5_lite.PipeLine(model_sequence, filelist, 'output_folder')
    pipe.run()
    pipe.all2mp3() #Not recommand to call this function if ouput sounds are used to train model.

main_pipe()
```
Every line in the model_sequence means the model name, the first separated file suffix, the second separated file suffix, and the input sound file suffix from last step.

Note that the first output file can either be vocals or be background, which depends on the model. 

## Why to write this?

- UVR5 is a little huge for one who already have pytorch environments. bs_roformer is the best weight to sperate vocal and instrument. While it needs a beta patch in UVR5. It is not so elegant to programer.

- It is ugly to use UVR5 with multi steps, like the output file will be like this:
```
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
```
Many 1_1_1 or 2_2_2 and () are displayed. And, it does not support batch operations: multi files and multi steps.

With this tool, the results will show in a better manner.
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2024/8/16     16:17        4229999 fly away_1_instrumental.mp3
-a----         2024/8/16     16:17        4229582 fly away_2_Karaoke_bg.mp3
-a----         2024/8/16     16:18        4229582 fly away_3_DeEcho_bg.mp3
-a----         2024/8/16     16:18        4229582 fly away_4_DeNoise_bg.mp3
-a----         2024/8/16     16:18        4229582 fly away_4_DeNoise_main.mp3
-a----         2024/8/16     16:17        4249644 小幸运_1_instrumental.mp3
-a----         2024/8/16     16:17        4249226 小幸运_2_Karaoke_bg.mp3
-a----         2024/8/16     16:17        4249226 小幸运_3_DeEcho_bg.mp3
-a----         2024/8/16     16:17        4249226 小幸运_4_DeNoise_bg.mp3
-a----         2024/8/16     16:17        4249226 小幸运_4_DeNoise_main.mp3
-a----         2024/8/16     16:17        3811204 魔鬼中的天使_1_instrumental.mp3
-a----         2024/8/16     16:17        3811204 魔鬼中的天使_2_Karaoke_bg.mp3
-a----         2024/8/16     16:17        3811204 魔鬼中的天使_3_DeEcho_bg.mp3
-a----         2024/8/16     16:17        3811204 魔鬼中的天使_4_DeNoise_bg.mp3
-a----         2024/8/16     16:17        3811204 魔鬼中的天使_4_DeNoise_main.mp3
```

## What skill shoud I have to use it?
- Simple Python skills. (Just modify model_sequence.)
- Senior Python environment install skills. (See the next section.)

## What is need to prepare?
- fp32 networks shown in the first section and put them in the models folder. fp16 models may not works but not tested. 
- A NVIDIA GPU with CUDA avaliable.
- Numpy and librosa are not matched in one place. Librosa still use `np.float` in utils.py, which is not allowed in latest numpy versions. Please change it to `float` manually.
- ffmpeg.exe and ffprobe.exe in environment PATH or the same folder with the python file you run.

## 原版UVR5参数说明（与本项目无关，单纯放着）
- segment size：选择分段大小以平衡速度、资源使用和质量。较小的尺寸消耗较少的资源。更大的规模消耗更多的资源，但可能会提供更好的结果。默认大小为256。质量可以根据您的选择而变化。
- overlap：此选项控制预测窗口之间的重叠量。更高的值可以提供更好的结果，但会导致更长的处理时间。对于非MDX23C型号：您可以在0.001-0.99之间进行选择。
- window size：选择窗口大小以平衡质量和速度：1024-快速但质量较差。512-中等速度和质量。320-需要更长的时间，但可能提供更好的质量。
- aggression setting：调整主干提取的强度：它的范围从-100到100。更大的值意味着更深的提取。通常，人声和乐器设置为5。值超过5可能会使非人声模型的声音变得浑浊。