import ffmpeg, os
import torch
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader

'''
原版UVR5说明：
segment size：选择分段大小以平衡速度、资源使用和质量。较小的尺寸消耗较少的资源。更大的规模消耗更多的资源，但可能会提供更好的结果。默认大小为256。质量可以根据您的选择而变化。
overlap：此选项控制预测窗口之间的重叠量。更高的值可以提供更好的结果，但会导致更长的处理时间。对于非MDX23C型号：您可以在0.001-0.99之间进行选择。
window size：选择窗口大小以平衡质量和速度：1024-快速但质量较差。512-中等速度和质量。320-需要更长的时间，但可能提供更好的质量。
aggression setting：调整主干提取的强度：它的范围从-100到100。更大的值意味着更深的提取。通常，人声和乐器设置为5。值超过5可能会使非人声模型的声音变得浑浊。

歌声分离提取：BS-Roformer-Viperx-1297 + 6_HP-Karaoke-UVR（5的话据说更激进） + UVR-De-Echo-(任一) + DeNoise（不必要）。
'''

model_names = ['1_HP-UVR.pth',
               '5_HP-Karaoke-UVR.pth',
               '6_HP-Karaoke-UVR.pth',
               'bs_roformer_ep_317_sdr_12.9755.ckpt',
               'UVR-De-Echo-Aggressive.pth',
               'UVR-De-Echo-Normal.pth',
               'UVR-DeEcho-DeReverb.pth',
               'UVR-DeNoise-Lite.pth',
               'UVR-DeNoise.pth']

def func(model_name, paths: list[str], sound_names: list[str] , root1='output', root2='output', suffix1='vocals', suffix2='instrumental'):
    assert len(paths) == len(sound_names)
    agg = 5
    model_path = 'models/'+model_name
    if 'bs_roformer' in model_name: 
        pre_fun = BsRoformer_Loader(model_path, device='cuda', is_half=False)
    elif 'De' in model_name:
        pre_fun = AudioPreDeEcho(agg, model_path, device='cuda', is_half=False)
    elif 'HP' in model_name:
        pre_fun = AudioPre(agg, model_path, device='cuda', is_half=False)
    for path, sound_name in zip(paths, sound_names):
        print(sound_name)
        info = ffmpeg.probe(path, cmd="ffprobe")
        if not (info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100"): 
            tmp_path = "%s/%s.reformatted.wav" % (os.path.join(os.environ["TEMP"]), os.path.basename(path))
            os.system(f'ffmpeg -i "{path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
            path = tmp_path
        pre_fun.run(path, sound_name, root1, root2, suffix1, suffix2, format='wav')

class PipeLine:
    def __init__(self, models: list[tuple[str]], soundfiles: list[str], output_path: str, soundnames = None): 
        self.models = models
        self.soundfiles = soundfiles
        self.ouput_path = output_path
        self.soundnames = soundnames if soundnames else self.generate_namelist()

    def generate_namelist(self):
        return [os.path.basename(filename).rsplit('.', 1)[0] for filename in self.soundfiles]
    
    def run(self):
        for modelname, suffix0, suffix1, previous_suffix in self.models:
            print(modelname)
            if previous_suffix is None: filelist = self.soundfiles
            else: filelist = [os.path.join(self.ouput_path, f'{sn}_{previous_suffix}.wav') for sn in self.soundnames]
            func(modelname, filelist, self.soundnames, self.ouput_path, self.ouput_path, suffix0, suffix1)

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def all2mp3(self, delete_wav=True): 
        '''
        Not recommand to call this function if ouput sounds are used to train model.
        '''
        for soundname in self.soundnames:
            for modelname, suffix0, suffix1, previous_suffix in self.models:
                for suffix in (suffix0, suffix1):
                    path = f'{self.ouput_path}/{soundname}_{suffix}'
                    os.system(f'ffmpeg -i {path}.wav -vn {path}.mp3 -q:a 2 -y')
                    if delete_wav: os.remove(path+'.wav')
            



def main():
    func('bs_roformer_ep_317_sdr_12.9755.ckpt', ['../resource/小幸运.mp3'], ['小幸运'])
    func('5_HP-Karaoke-UVR.pth', ['output/小幸运_vocals.wav'], ['小幸运'], suffix1='Karaoke_bg', suffix2='Karaoke_main')
    func('UVR-De-Echo-Normal.pth', ['output/小幸运_Karaoke_main.wav'], ['小幸运'], suffix1='DeEcho_main', suffix2='DeEcho_bg')
    func('UVR-DeNoise.pth', ['output/小幸运_DeEcho_main.wav'], ['小幸运'], suffix1='DeNoise_bg', suffix2='DeNoise_main')

def main_pipe():
    model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', 'vocals', 'instrumental', None),
                     ('5_HP-Karaoke-UVR.pth', 'Karaoke_bg', 'Karaoke_main', 'vocals'),
                     ('UVR-De-Echo-Normal.pth', 'DeEcho_main', 'DeEcho_bg', 'Karaoke_main'),
                     ('UVR-DeNoise.pth', 'DeNoise_bg', 'DeNoise_main', 'DeEcho_main')]                  
                  
    filelist = [f'../resource/{i}.mp3' for i in ('魔鬼中的天使', '小幸运')]
    pipe = PipeLine(model_sequence, filelist, 'output')
    pipe.run()
    pipe.all2mp3()

main_pipe()
