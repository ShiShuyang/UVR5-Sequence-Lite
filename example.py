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

def main_pipe():
    model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', '1_vocals', '1_instrumental', None),
                     ('5_HP-Karaoke-UVR.pth', '2_Karaoke_bg', '2_Karaoke_main', '1_vocals'),
                     ('UVR-De-Echo-Normal.pth', '3_DeEcho_main', '3_DeEcho_bg', '2_Karaoke_main'),
                     ('UVR-DeNoise.pth', '4_DeNoise_bg', '4_DeNoise_main', '3_DeEcho_main')]                  
                  
    filelist = [f'../resource/{i}.mp3' for i in ('魔鬼中的天使', '小幸运', 'fly away')]
    pipe = UVR5_lite.PipeLine(model_sequence, filelist, 'output_folder')
    pipe.run(delete_last_step=True)
    pipe.all2mp3(delete_wav=True) #Not recommand to call this function if ouput sounds are used to train model.

main_pipe()
