import ffmpeg, os
import torch
#from mdxnet import MDXNetDereverb
from .vr import AudioPre, AudioPreDeEcho
from .bsroformer import BsRoformer_Loader

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
        print(sound_name, path)
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
    
    def run(self, delete_last_step = True):
        for modelname, suffix0, suffix1, previous_suffix in self.models:
            print(modelname)
            if previous_suffix is None: filelist = self.soundfiles
            else: filelist = [os.path.join(self.ouput_path, f'{sn}_{previous_suffix}.wav') for sn in self.soundnames]
            func(modelname, filelist, self.soundnames, self.ouput_path, self.ouput_path, suffix0, suffix1)
            if delete_last_step and previous_suffix:
                for filename in filelist: os.remove(filename)

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def all2mp3(self, delete_wav=True): 
        '''
        Not recommand to call this function if ouput sounds are used to train model.
        '''
        for soundname in self.soundnames:
            for modelname, suffix0, suffix1, previous_suffix in self.models:
                for suffix in (suffix0, suffix1):
                    path = f'{self.ouput_path}/{soundname}_{suffix}'
                    if os.path.isfile(path+'.wav'):
                        os.system(f'ffmpeg -i "{path}.wav" -vn "{path}.mp3" -q:a 2 -y')
                        if delete_wav: os.remove(path+'.wav')
