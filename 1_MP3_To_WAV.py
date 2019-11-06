from pydub import AudioSegment
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('디렉토리를 만들 수 없습니다.')

createFolder("./Image/")
createFolder("./Image/Happy(tension up)/")
createFolder("./Image/Sad(이별 및 슬픔)/")

# createFolder("./Music")
# createFolder("./Music/MP3")
# createFolder("./Music/MP3/Happy(tension up)")
# createFolder("./Music/MP3/Sad(이별 및 슬픔)")
# MP3폴더에는 수동으로 지우기! (애초에 파일을 넣어야 하기 때문에 폴더가 존재해야하여야함)

createFolder("./Music/WAV")
createFolder("./Music/WAV/Happy(tension up)")
createFolder("./Music/WAV/Sad(이별 및 슬픔)")

createFolder("./test_set/")

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