import re

from math import ceil
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import glob

def collage_maker(frame_width, images_per_row, padding, path, save_name, digits_to_remove=1):
    resizer = transforms.Resize(256)

    images = glob.glob('{}/*.jpg'.format(path))
    images = sorted(images, key=lambda x:int(re.findall("(\d+)",x)[-1]), reverse=False)

    font = ImageFont.truetype('Arial.ttf', 20)

    img_width, img_height = 256, 256
    sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)       #scaling factor
    scaled_img_width = ceil(img_width*sf)                   #s
    scaled_img_height = ceil(img_height*sf)

    number_of_rows = ceil(len(images)/images_per_row)
    frame_height = ceil(sf*img_height*number_of_rows) 

    new_im = Image.new('RGB', (frame_width, frame_height), color=(230, 230, 230))

    position = (0,0)

    i,j=0,0
    for num, im in enumerate(images):
        try:
            if num % images_per_row == 0:
                i = 0
            im = Image.open(im)
        except IOError:
            print(f"Error opening image: {im}")
            im = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        else:
            # convert to rgb
            im_rgb = im.convert('RGB')
            idx = ''.join(list(filter(str.isdigit, im.filename))[digits_to_remove:])
            im = resizer(im_rgb)

        #Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width, scaled_img_height))
        draw = ImageDraw.Draw(im)
        bbox = draw.textbbox(position, idx, font=font)
        draw.rectangle(bbox, fill=(1, 1, 1))
        draw.text(position, idx, fill=(255, 255, 255), font=font)
        #Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j // images_per_row) * scaled_img_height
        new_im.paste(im, (i, y_cord))
        # print(i, y_cord)
        i = (i + scaled_img_width) + padding
        j += 1
            
    new_im.save(save_name, 'JPEG', quality=80, optimize=True, progressive=True)

# real_path_gossipcop = '././data/gossipcop/news_images_en/real'
# fake_path_gossipcop = '././data/gossipcop/news_images_en/fake'

# collage_maker(1000, 5, 40, fake_path_gossipcop, './data_analysis/collage_fake_gossipcop.jpg', 0)
# collage_maker(1000, 5, 40, real_path_gossipcop, './data_analysis/collage_real_gossipcop.jpg', 0)

# real_path = '././data/politifact_v2/news_images/real'
# fake_path = '././data/politifact_v2/news_images/fake'

# collage_maker(1000, 5, 40, fake_path, './data_analysis/collage_fake_politifact_v2.jpg', 2)
# collage_maker(1000, 5, 40, real_path, './data_analysis/collage_real_politifact_v2.jpg', 2)

# with_web_archive_path = '././data/images_with_web_archive_url'
# without_web_archive_path = '././data/images_without_web_archive_url'

# collage_maker(1000, 5, 40, with_web_archive_path, './data_analysis/collage_with_web_archive.jpg')
# collage_maker(1000, 5, 40, without_web_archive_path, './data_analysis/collage_without_web_archive.jpg')


# uncleaned_path_real = '././data/politifact_v2/news_images_uncleaned/real'
# uncleaned_path_fake = '././data/politifact_v2/news_images_uncleaned/fake'

# collage_maker(1000, 5, 40, uncleaned_path_real, './data_analysis/collage_real_uncleaned_politifact_v2.jpg', 1)
# collage_maker(1000, 5, 40, uncleaned_path_fake, './data_analysis/collage_fake_uncleaned_politifact_v2.jpg', 1)


# invalid_path_real = './data/politifact_v2/data_presentation/invalid/real'
# invalid_path_fake = './data/politifact_v2/data_presentation/invalid/fake'

# collage_maker(300, 1, 40, invalid_path_real, './data_analysis/collage_real_invalid_politifact_v2.jpg', 1)
# collage_maker(300, 1, 40, invalid_path_fake, './data_analysis/collage_fake_invalid_politifact_v2.jpg', 1)

# typical_path_real = './data/politifact_v2/data_presentation/typical/real'
# typical_path_fake = './data/politifact_v2/data_presentation/typical/fake'

# collage_maker(300, 2, 40, typical_path_real, './data_analysis/collage_real_typical_politifact_v2.jpg', 1)
# collage_maker(300, 2, 40, typical_path_fake, './data_analysis/collage_fake_typical_politifact_v2.jpg', 1)

# typical_path_fake_g = './data/gossipcop/data_presentation/typical/fake'
# typical_path_real_g = './data/gossipcop/data_presentation/typical/real'

# collage_maker(300, 2, 40, typical_path_fake_g, './data_analysis/collage_fake_typical_gossipcop.jpg', 0)
# collage_maker(300, 2, 40, typical_path_real_g, './data_analysis/collage_real_typical_gossipcop.jpg', 0)

real_path = '././data/politifact_v4/news_images/real'
fake_path = '././data/politifact_v4/news_images/fake'

collage_maker(1000, 5, 40, fake_path, './data_analysis/collage_fake_politifact_v4.jpg', 1)
collage_maker(1000, 5, 40, real_path, './data_analysis/collage_real_politifact_v4.jpg', 1)

real_path = '././data/gossipcop_v4/news_images/real'
fake_path = '././data/gossipcop_v4/news_images/fake'

collage_maker(1000, 5, 40, fake_path, './data_analysis/collage_fake_gossipcop_v4.jpg', 1)
collage_maker(1000, 5, 40, real_path, './data_analysis/collage_real_gossipcop_v4.jpg', 1)







