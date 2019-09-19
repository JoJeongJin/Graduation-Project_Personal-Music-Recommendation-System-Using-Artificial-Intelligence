from pydub import AudioSegment
import os

music_type = ["negative", "positive"]

mp3_path_directory = "./Music/MP3/"

for _list in music_type:
    file_list_dir = mp3_path_directory + _list +"/"
    file_list = os.listdir(file_list_dir)
    print(file_list)
    for item in file_list:
        sound = AudioSegment.from_mp3(file_list_dir+item)

        print("./Music/WAV/"+_list+"/")
        sound.export("./Music/WAV/"+_list+"/"+item+".wav", format="wav")