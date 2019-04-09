# setting
```
pip install -r requirements.txt
```
# dataset 구성
앵커브리핑 오디오 다운로드,  wav 파일 분할 등 작업을 위한 코드는
https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
이곳에 있는 코드를 활용. 

tacotron에서 " carpedm20"님은 Google Speech Recognition API 결과가 썩 좋지는 못하기 때문에, alignment.py에서, 원문을 참고하여 수정하려는 시도

데이타를 받깅 위한 패키지는 위 명령어로 실행된다. 그러나 ffmpeg 패키지는 잘 설치해야한다.(윈도우에서 성공, 리눅스에서 실패)

ffmpeg 에러시 video, asset, audio(영상, 대본, 음성) 중 audio가 추출되지 않는다.

news_ids.json을 이용하여 목록을 만들고 아래와 같이 datasets 밑에 son 밑에 받는다.

├─datasets

│  └─son

│     ├─assets

│     ├─audio

│     └─video

1) "news_ids.json"을 datasets/son 아래에 복사
  - "news_ids.json"이 없으면 web상에 있는 앵커브리핑 자료로부터 새로 만들기 때문에 다른 목록이 만들어짐
  
2) 다운로드 코드 실행, 시간이 오래 걸림, video, audio, assets 파일 생성, 743개 파일 22G
  - ffmpeg 오류시 audio 추출 안됨
```
python -m datasets.son.download_custom
```

3) wav 파일을 침묵구간 기준으로 쪼갬, 시간 많이 소요(2~3시간)
  - 완료시 audio 파일에 wav파일이 잘게 나누어 져있음, 앵커브리핑 1회분 평균 55개
  - 중간에 오류발생시 원본파일 빼고 삭제 후 다시
  - wav파일 path는 환경에 맞게 수정하면 됨
```
python -m audio.silence --audio_pattern "./datasets/son/audio/*.wav" --method=pydub
```

4) 스크립트 교정 작업

  - assets에 있는 원문과 Google Speech Recognition API로 사전작업한 "AnchorBriefing-recognition.json" 파일을 참고하여
  
  - audio 디렉토리에 있는 wav 파일을 직접 들으면서 "AnchorBriefing-recognition.json" 수정
  
  - 앵커브리핑 1회 수정시 약 10~15분 소요
  
  - 작업시 주의할점
      1. 쉼표, 따옴표, 마침표는 모두 제거한다.
      2. 숫자 읽기 : 8명 -> 여덞명
      3. 인터뷰 : 앵커 목소리가 아닌 다른사람 목소리 시 작업할 필요 없음. json 에서 제거
      4. 버벅거리는 부분은 들리는데로(예..어..저 같은말들)
      5. 영어 : 영어인 경우 한글로(Don't worry be happy -> 돈워리 비 해피)
      6. 띄어쓰기 : 맞춤법에 맞게
    
   
    
    
    
# Multi-Speaker Tocotron2 + Wavenet Vocoder + Korean TTS
Tacotron2 모델과 Wavenet Vocoder를 결합하여  한국어 TTS구현하는 project입니다.
Tacotron2 모델을 Multi-Speaker모델로 확장했습니다.

Based on 
- https://github.com/keithito/tacotron
- https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
- https://github.com/Rayhane-mamah/Tacotron-2
- https://github.com/hccho2/Tacotron-Wavenet-Vocoder


## Tacotron 2
- Tacotron 모델에 관한 설명은 이전 [repo](https://github.com/hccho2/Tacotron-Wavenet-Vocoder) 참고하시면 됩니다.
- [Tacotron2](https://arxiv.org/abs/1712.05884)에서는 모델 구조도 바뀌었고, Location Sensitive Attention, Stop Token, Vocoder로 Wavenet을 제안하고 있다.
- Tacotron2의 대표적인 구현은 [Rayhane-mamah](https://github.com/Rayhane-mamah/Tacotron-2)입니다. 이 역시, [keithito](https://github.com/keithito/tacotron), [r9y9](https://github.com/r9y9/wavenet_vocoder)의 코드를 기반으로 발전된 것이다.

## This Project
* Tacotron2 모델로 한국어 TTS를 만드는 것이 목표입니다.
* [Rayhane-mamah](https://github.com/Rayhane-mamah/Tacotron-2)의 구현은 Customization된 Layer를 많이 사용했는데, 제가 보기에는 너무 복잡하게 한 것 같아, Cumomization Layer를 많이 줄이고, Tensorflow에 구현되어 있는 Layer를 많이 활용했습니다.
* teacher forcing 방식의 train sample은 2000 step부터, free forcing 방식의 test sample은 3000 step부터 알아들을 수 있는 정도의 음성을 만들기 시작합니다.
## 단계별 실행

### 실행 순서
- Data 생성: 한국어 data의 생성은 이전 [repo](https://github.com/hccho2/Tacotron-Wavenet-Vocoder) 참고하시면 됩니다.
- 생성된 Data는 아래의 'data_paths'에 지정하면 된다.
- tacotron training 후, synthesize.py로 test.
- wavenet training 후, generate.py로 test(tacotron이 만들지 않은 mel spectrogram으로 test할 수도 있고, tacotron이 만든 mel spectrogram을 사용할 수도 있다.)
- 2개 모델 모두 train 후, tacotron에서 생성한 mel spectrogram을 wavent에 local condition으로 넣어 test하면 된다.


### Tacotron2 Training
- train_tacotron2.py 내에서 '--data_paths'를 지정한 후, train할 수 있다. data_path는 여러개의 데이터 디렉토리를 지정할 수 있습니다.
```
parser.add_argument('--data_paths', default='.\\data\\moon,.\\data\\son')
```
- train을 이어서 계속하는 경우에는 '--load_path'를 지정해 주면 된다.
```
parser.add_argument('--load_path', default='logdir-tacotron2/moon+son_2019-02-27_00-21-42')
```

- model_type은 'single' 또는 ' multi-speaker'로 지정할 수 있다. speaker가 1명 일 때는, hparams의 model_type = 'single'로 하고 train_tacotron2.py 내에서 '--data_paths'를 1개만 넣어주면 된다.
```
parser.add_argument('--data_paths', default='D:\\Tacotron2\\data\\moon')
```
- 하이퍼파라메터를 hparmas.py에서 argument를 train_tacotron2.py에서 다 설정했기 때문에, train 실행은 다음과 같이 단순합니다.
> python train_tacotron2.py
- train 후, 음성을 생성하려면 다음과 같이 하면 된다. '--num_speaker', '--speaker_id'는 잘 지정되어야 한다.
> python synthesizer.py --load_path logdir-tacotron2/moon+son_2019-02-27_00-21-42 --num_speakers 2 --speaker_id 0 --text "오스트랄로피테쿠스 아파렌시스는 멸종된 사람족 종으로, 현재에는 뼈 화석이 발견되어 있다." 



### Wavenet Vocoder Training
- train_vocoder.py 내에서 '--data_dir'를 지정한 후, train할 수 있다.
- memory 부족으로 training 되지 않거나 너무 느리면, hyper paramerter 중 sample_size를 줄이면 된다. 물론 batch_size를 줄일 수도 있다.
```
DATA_DIRECTORY =  'D:\\Tacotron2\\data\\moon,D:\\Tacotron2\\data\\son'
parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory containing data')
```
- train을 이어서 계속하는 경우에는 '--logdir'를 지정해 주면 된다.
```
LOGDIR = './/logdir-wavenet//train//2018-12-21T22-58-10'
parser.add_argument('--logdir', type=str, default=LOGDIR)
```
- wavenet train 후, tacotron이 생성한 mel spectrogram(npy파일)을 local condition으로 넣어서 STT의 최종 결과를 얻을 수 있다.
> python generate.py --mel ./logdir-wavenet/mel-moon.npy --gc_cardinality 2 --gc_id 0 ./logdir-wavenet/train/2018-12-21T22-58-10

### Result
- Tacotron의 batch_size = 32, Wavenet의 batch_size=8. GTX 1080ti.
- Tacotron은 step 100K, Wavenet은 177K 만큼 train.
- samples 디렉토리에는 생성된 wav파일이 있다.
- Griffin-Lim으로 생성된 것과 Wavenet Vocoder로 생성된 sample이 있다.
- Wavenet으로 생성된 음성은 train 부족으로 잡음이 섞여있다.


