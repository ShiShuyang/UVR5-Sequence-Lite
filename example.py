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
    model_sequence = [('bs_roformer_ep_317_sdr_12.9755.ckpt', 'vocals', 'instrumental', None),
                     ('5_HP-Karaoke-UVR.pth', 'Karaoke_bg', 'Karaoke_main', 'vocals'),
                     ('UVR-De-Echo-Normal.pth', 'DeEcho_main', 'DeEcho_bg', 'Karaoke_main'),
                     ('UVR-DeNoise.pth', 'DeNoise_bg', 'DeNoise_main', 'DeEcho_main')]                  
                  
    filelist = [f'../resource/{i}.mp3' for i in ('魔鬼中的天使', '小幸运')]
    pipe = UVR5_lite.PipeLine(model_sequence, filelist, 'output_folder')
    pipe.run()
    pipe.all2mp3()

main_pipe()
