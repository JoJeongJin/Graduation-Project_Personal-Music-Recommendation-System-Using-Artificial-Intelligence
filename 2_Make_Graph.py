import os
from Make_Music_Graph import make_graph

music_type = ["negative", "positive"]

wav_path_directory = "./Music/WAV/"
img_path = "./Image/"

for _list in music_type:
    file_list_dir = wav_path_directory + _list +"/"
    file_list = os.listdir(file_list_dir)
    print(file_list)
    print("파일 개수: "+ str(len(file_list)))
    for item in file_list:
        print(item)
        make_graph( (img_path+_list+"/"),"./Music/WAV/"+_list+"/"+item,item)

