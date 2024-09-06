import UVR5_lite

model_names = ['1_HP-UVR.pth',
               '5_HP-Karaoke-UVR.pth',
               '6_HP-Karaoke-UVR.pth',
               'bs_roformer_ep_317_sdr_12.9755.ckpt',
               'UVR-De-Echo-Aggressive.pth',
               'UVR-De-Echo-Normal.pth',
               'UVR-DeEcho-DeReverb.pth',
               'UVR-DeNoise-Lite.pth',
               'UVR-DeNoise.pth']

def main_func():
    UVR5_lite.func('bs_roformer_ep_317_sdr_12.9755.ckpt', ['../resource/music/小幸运.mp3'], ['小幸运'], 'output_folder', 'output_folder')
    UVR5_lite.func('5_HP-Karaoke-UVR.pth', ['output_folder/小幸运_vocals.wav'], ['小幸运'], 'output_folder', 'output_folder', suffix1='Karaoke_bg', suffix2='Karaoke_main')
    UVR5_lite.func('UVR-De-Echo-Normal.pth', ['output_folder/小幸运_Karaoke_main.wav'], ['小幸运'], 'output_folder', 'output_folder', suffix1='DeEcho_main', suffix2='DeEcho_bg')
    UVR5_lite.func('UVR-DeNoise.pth', ['output_folder/小幸运_DeEcho_main.wav'], ['小幸运'], 'output_folder', 'output_folder', suffix1='DeNoise_bg', suffix2='DeNoise_main')

def main_pipe():
    '''
    Recommand
    '''
    model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', '1_vocals', '1_instrumental', None),
                     ('5_HP-Karaoke-UVR.pth', '2_Karaoke_bg', '2_Karaoke_main', '1_vocals'),
                     ('UVR-De-Echo-Normal.pth', '3_DeEcho_main', '3_DeEcho_bg', '2_Karaoke_main'),
                     ('UVR-DeNoise.pth', '4_DeNoise_bg', '4_DeNoise_main', '3_DeEcho_main')]                  
                  
    filelist = [f'../resource/music/{i}.mp3' for i in ('魔鬼中的天使', '小幸运', 'fly away')]
    pipe = UVR5_lite.PipeLine(model_sequence, filelist, 'output_folder')
    pipe.run(delete_last_step=True)
    pipe.all2mp3(delete_wav=True) #Not recommand to call this function if ouput sounds are used to train model.

main_pipe()
