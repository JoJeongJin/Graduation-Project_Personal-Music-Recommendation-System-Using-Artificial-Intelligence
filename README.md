"CNN 음악 감정 분류기" - CNN을 이용하여 어떤 감정의 음악인지 분류해주는 프로그램

이 프로젝트를 하기 전에 Add_Folder_List.PNG에 있는 폴더들을 만들어야 합니다.

단계는 다음과 같습니다.
1. MP3_To_WAV.py - 가지고 있는 MP3파일을 WAV로 바꿔주는 코드
2. Make_Graph.py - WAV파일을 그래프 이미지로 바꿔주는 코드
3. Image_resize.py - 그래프 이미지 가지고 있는것을 정사각형으로 만들어주는 코드 (Train에서 쓰기 위함)
4. Train.py - 가지고 있는 Positive 이미지와 Negative 이미지를 통해 학습시키는 코드
5. Test.py - Test_set에 설정해놓은 파일들을 이용해 실제로 학습이 잘되었는지 테스트 해보는 코드
이런 순서로 실행하면 됩니다.

