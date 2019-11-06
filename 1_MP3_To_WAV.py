from pydub import AudioSegment
import os

music_type = ["Happy(tension up)", "Sad(이별 및 슬픔)"]

mp3_path_directory = "./Music/MP3/"

for _list in music_type:
    file_list_dir = mp3_path_directory + _list +"/"
    file_list = os.listdir(file_list_dir)
    print(file_list)
    print("파일 개수: "+ str(len(file_list)))
    for item in file_list:
        sound = AudioSegment.from_mp3(file_list_dir+item)
        print(item)
        sound.export("./Music/WAV/"+_list+"/"+item+".wav", format="wav")