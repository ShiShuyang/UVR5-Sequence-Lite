import UVR5_lite, ssytool as ssy

model_names = ['1_HP-UVR.pth',
               '5_HP-Karaoke-UVR.pth',
               '6_HP-Karaoke-UVR.pth',
               'bs_roformer_ep_317_sdr_12.9755.ckpt',
               'UVR-De-Echo-Aggressive.pth',
               'UVR-De-Echo-Normal.pth',
               'UVR-DeEcho-DeReverb.pth',
               'UVR-DeNoise-Lite.pth',
               'UVR-DeNoise.pth']

#音视频分离：ffmpeg -i 潘涛.mp4 -vn -acodec copy 潘涛.m4a

def main_pipe():

    model_sequence = [('5_HP-Karaoke-UVR.pth', 'Karaoke_bg', 'Karaoke_main', None),
                      ('UVR-DeNoise.pth', 'DeNoise_bg', 'DeNoise_main', 'Karaoke_main')]                  

    filelist = [i for i in ssy.oswalk('../resource/clean_voice/LiZhu') if i.endswith('wav')]
    print(filelist)
    pipe = UVR5_lite.PipeLine(model_sequence, filelist, '../resource/clean_voice/LiZhu2')
    pipe.run(delete_last_step=True)
    #pipe.all2mp3(delete_wav=True) #Not recommand to call this function if ouput sounds are used to train model.

main_pipe()
