import os
from PIL import Image

music_type = ["Happy(tension up)", "Sad(이별 및 슬픔)", "Soso(약간 잠자기 전에 듣기 좋은 노래)"]

img_path = "./Image/"

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


