import os
from PIL import Image

music_type = ["negative", "positive"]

img_path =  "./Image/"

for _list in music_type:
    file_list_dir = img_path + _list +"/"
    file_list = os.listdir(file_list_dir)
    print(file_list)
    for item in file_list:
        print(item)
        image = Image.open(file_list_dir+item)
        resize_image = image.resize((512, 512))
        resize_image.save(file_list_dir+item)
        img = Image.open(file_list_dir+item).convert('LA')
        img.save(file_list_dir+item)


