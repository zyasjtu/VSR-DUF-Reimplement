import os

img_dir = './result/'

if __name__ == '__main__':
    for vdo_name in os.listdir(img_dir):
        print(img_dir + vdo_name)
        os.system('ffmpeg -i ' + img_dir + vdo_name + '/%5d.bmp -pix_fmt yuv420p -vsync 0 ' + img_dir + vdo_name + '_Res.y4m')
        os.system('ffmpeg -i ' + img_dir + vdo_name + '_Res.y4m -vf select=\'not(mod(n\\,25))\' -vsync 0  -y ' + img_dir + vdo_name + '_Sub25_Res.y4m')
