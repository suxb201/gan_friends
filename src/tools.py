# 使用 epoch 数生成单张图片
import imageio
import PIL
import glob


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


anim_file = 'dcgan_imcy.gif'

with imageio.get_writer(anim_file, mode='I', duration=0.5) as writer:
    filenames = glob.glob('imcy/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
