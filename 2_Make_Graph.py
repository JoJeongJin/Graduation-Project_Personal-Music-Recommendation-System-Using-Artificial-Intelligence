import os
from Make_Music_Graph import make_graph

music_type = ["Happy(tension up)", "Sad(이별 및 슬픔)", "Soso(약간 잠자기 전에 듣기 좋은 노래)"]

wav_path_directory = "./Music/WAV/"
img_path = "./Image/"

index = 0

for _list in music_type:
    file_list_dir = wav_path_directory + _list +"/"
    file_list = os.listdir(file_list_dir)
    print(file_list)
    print("파일 개수: "+ str(len(file_list)))
    for item in file_list:
        print(item)
        index = index + 1
        make_graph( (img_path+_list+"/"),"./Music/WAV/"+_list+"/"+item, item, _list, index)

