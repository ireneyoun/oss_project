<div align= "center">
    <img src="https://capsule-render.vercel.app/api?type=shark&color=0:f0d1d1,300:d6edff&height=250&text=세상을%20바꾸는%20고양이%20🐈&animation=fadeIn&fontColor=000000&fontSize=60" />
    </div>
    
# SMWU 24-1 오픈소스프로그래밍 기말 프로젝트

:heart_eyes_cat: 2313621 윤다연

    - 이상행동 영상 데이터 수집 및 전처리
    
    - 영상 기반 이상행동 종류 분류 모델 자료 수집 및 학습

:smile_cat: 2210801 선영주

    - 이메일 알림 발송 모듈 설계 및 개발

    - 웹캠 구현 및 실행

:pouting_cat: 2210888 송민지

    - 데이터셋 분할(훈련, 테스트, 검증 데이터셋)

    - 영상 기반 이상행동 감지

    - 학습된 모델 성능 평가 (영상)

## GitHub ID
|<img src="https://avatars.githubusercontent.com/u/144865717?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/165630285?v=4" width="150" height="150"/>|<img src="https://avatars.githubusercontent.com/u/165617182?v=4" width="150" height="150"/>|
|:-:|:-:|:-:|
|윤다연<br/>[@ireneyoun](https://github.com/ireneyoun)|선영주<br/>[@dudwntjs](https://github.com/dudwntjs)|송민지<br/>[@ssong721](https://github.com/ssong721)|

-----------
## 개요 및 개발 필요성
- 기술개발 배경

    - 현대 사회에서 범죄율 증가와 안전에 대한 우려가 높아짐에 따라, 보다 효과적인 보안 시스템의 필요성이 증대되고 있음.
  
    - CCTV 카메라는 보편화되어 있으나, 대부분의 CCTV 시스템은 수동 모니터링에 의존하고 있어, 실시간 범죄 감지에는 한계가 존재
      
    - 정부에서는 이러한 범죄를 예방하기 위해 AI CCTV에 관심을 가지고 2023년 서울시에서 '폭력행동 감지' 지능형 CCTV를 설치하기 시작
      
    - 하지만 일반 가게에 설치하기에는 비용 부담과 정확도 증명의 문제
      
    - 이를 해결하기 위해 <ins>CCTV 실시간 범죄 분석 및 보안 알림</ins>이 중요한 기술적 대안
      
<p align="center">
    <figure class="half">
             <p align="center"><a href="link"><img width="300" alt="서울시, '폭력행동 감지' 지능형 CCTV"         src="https://github.com/ireneyoun/oss_project/assets/165630285/50b9f164-02d2-4be7-9c40-c4bbd852ee67" >
            <a href="link"><img width="400" alt="'묻지마 범죄 막는다' 2026년까지 서울 전역 지능형 CCTV로 교체" src="https://github.com/ireneyoun/oss_project/assets/165630285/7438a7cf-fc02-45d7-a056-b46be508b2b4">
    </figure>
</p>
 
- 기술개발 목표

    - <ins>실시간으로 CCTV 영상을 분석</ins>하여 범죄 행위(폭력)를 신속하게 감지한 후, 가게 주인에게 <ins>즉각적으로 알림</ins>을 제공하여 빠른 대응이 가능하도록 함.
 
- 기대효과
  
    **[기술적 기대효과]**
    - 인공지능과 머신러닝 알고리즘을 사용하여 영상 분석의 정확도와 효율성을 크게 향상시킬 수 있음.
 
    - 실시간 감지 및 알림 시스템으로 인해 가게 내 범죄율이 현저히 감소할 것이며, 가게 주인 및 보안 담당자가 신속하게 대응하여 범죄 상황을 조기에 해결 가능함.
 
    - 비디오 데이터 분석 기술의 객체 인식 능력 향상

    **[사회적 기대효과]**
    - 실시간 감지 및 알림 시스템으로 인해 가게 내 범죄율이 현저히 감소
      
    - 범죄 행위에 대한 명확한 증거를 제공하여 법적 보호를 강화
      
    - 가게 주인 및 보안 담당자가 신속하게 대응하여 범죄 상황을 조기에 해결 가능
      
    - 실시간 데이터 분석 및 개발 경험을 통한 우수 개발 인력 양성

-----------
## 개발환경

|**필수 라이브러리 및 패키지**|**버전**|**설명**|
|------|---|---|
|Python|3.x|프로젝트를 실행하기 위한 기본 프로그래밍 언어 버전|
|OpenCV(cv2)|4.5.x|이미지 및 비디오 처리에 특화된 라이브러리|
|xml.etree.ElementTree (ET)||XML 파일을 파싱하기 위해 필요한 라이브러리|
|PyTorch|1.10.x|딥 러닝 연구 및 개발을 위한 오픈소스 머신 러닝 라이브러리|
|numpy|1.21.x|수치 계산을 위한 파이썬 라이브러리|
|ONNXRuntime|1.10.x|ONNX 모델의 실행을 위한 고성능 런타임|

-----------
## 데이터셋
- 사람 이상 행동 데이터

    - 바운딩 박스로 라벨링이 되어 있음
      
    - 오픈소스 프로그래밍 수업 실습에서 얼굴 탐지를 할 때 네모 박스를 표시했던 것과 동일한 방식

      https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71550

<p align="center">
<img width="900" alt="스크린샷 2024-06-20 03 20 06" src="https://github.com/ireneyoun/oss_project/assets/165630285/a55cf3f8-b758-4c8c-ac74-e3e0238ea975">
</p>

-----------
## 데이터 전처리
- OpenCV를 이용한 비디오 처리 및 객체 추적을 활용

- Roboflow를 통해 데이터셋을 모델 학습에 적합한 형식으로 자동 변환

### 목표
: 비디오와 해당 XML 라벨을 처리

### 주요 기능
1. 비디오 전처리
 
    - OpenCV를 사용하여 비디오 파일로부터 프레임 단위로 이미지를 추출하고 저장

    - 각 프레임을 이미지 파일로 저장

<p align="center">
<img width="600" alt="image" src="https://github.com/ireneyoun/oss_project/assets/165630285/1aae9d95-5b0b-415d-8596-81ae5fc32a7f">
</p>

2. 라벨링
   
- 데이터 증강(Augmentation)
  
    - 모델 학습에 필요한 데이터를 변형
      
- robflow의 효과적인 라벨링 도구를 사용
  
    - 객체의 바운딩 박스를 지정하고, 라벨을 추가하여 효율적으로 데이터셋을 준비

<p align="center">
<img width="600" alt="image" src="https://github.com/ireneyoun/oss_project/assets/165630285/fefd994d-8f8e-4729-8987-f8a116fe4875">
</p>
      
3.	데이터셋 처리

    - XML 파일 디렉토리 내의 모든 XML 파일을 처리하여 관련 비디오 파일 추적 및 분류
      
4.	객체 추적기 초기화

    - CSRT(CSR-DCF) 객체 추적기를 초기화하여 객체를 추적
      
5.	바운딩 박스
    - 객체 추적기를 사용하여 영상에서 객체의 위치를 추적하고 바운딩 박스 생성
  
    - 객체 위치를 시각적으로 표시하여, 이상 행동 감지에 활용

-----------
## 모델 학습
- Yolov5를 구현하고 Roboflow를 사용하여 데이터셋을 관리
  
    - Yolov5 : 실시간 객체 감지를 위한 딥러닝 프레임워크
      
    - Roboflow : 데이터셋 관리 및 전처리 도구

### 목표
: Yolov5와 Roboflow를 사용하여 객체 탐지 모델을 쉽게 학습시키고, 다양한 플랫폼에서 사용할 수 있도록 ONNX 포맷으로 변환

### 주요 기능
1. Yolov5 설치 및 설정
    - Yolov5 GitHub 저장소를 클론하고 필요한 라이브러리를 설치

2. Roboflow 데이터셋 관리
    - Roboflow를 사용하여 객체 탐지 모델 학습에 필요한 데이터셋을 다운로드하고 설정
      
3. 모델 학습
   - Yolov5를 사용하여 객체 탐지 모델을 학습
  
4. 모델 평가
    - 학습된 모델을 사용하여 이미지에서 객체를 탐지하고 결과를 시각화

<figure class="half">
             <p align="center"><a href="link"><img width="300" alt="image" src="https://github.com/ireneyoun/oss_project/assets/165630285/21ec9d97-f0ab-4261-bced-a1fd0c708157">
 <a href="link"><img width="300" alt="image" src="https://github.com/ireneyoun/oss_project/assets/165630285/b52d9949-03e1-481c-b259-49c53d26a826">
 </figure>
</p>

5. ONNX 변환
    -  학습된 모델을 ONNX 포맷으로 변환

-----------
## 구현
: 웹캠으로 실시간 비디오 영상을 받아 이상 행동("abnormal")을 감지하는 모델을 실행하고, 감지되면 이메일 알림을 보내는 기능을 제공
1. 실시간 비디오
    - 웹캠을 통해 실시간으로 영상을 입력 받음

2. 객체 감지
    - ONNX 모델을 사용하여 객체를 감지하고, 비정상 행동을 식별
      
3. 이메일 알림
   - 비정상 행동이 감지되면 지정된 이메일 주소로 알림
   - Gmail SMTP 설정 사용
  
4. 결과 시각화
    - 감지된 객체를 영상 위에 표시하여 시각적 확인 가능

-----------
## 실행 방법
### test용 비디오

🔗 https://drive.google.com/drive/folders/1sgaSJWEwlKMIIOLRbBqDNCZCqqHvQPhy?usp=sharing

: 해당 링크에 있는 비디오를 다운 받아 yolov5-master 폴더 안에 저장한 후 테스트 가능

1. 지정한 비디오로 테스트
   - github 소스를 다운 받은 후 yolo5s_fight.onnx 파일을 yolov5-master 폴더 안에 넣고 위의 test용 비디오를 yolov5-master 폴더 안에 저장.
   - Anaconda prompt에서 밑에 나와있는 코드를 입력하면 실행 가능
   - 밑 코드에서 test.mp4 부분을 테스트하고자하는 영상의 이름으로 변경

    python detect_update.py --source test.mp4 --weights yolo5s_fight.onnx

2. 웹캠으로 테스트
   - github 소스를 다운 받은 후 yolo5s_fight.onnx 파일을 yolov5-master 폴더 안에 넣고 위의 test용 비디오를 yolov5-master 폴더 안에 저장.
   - Anaconda prompt에서 밑에 나와있는 코드를 입력하면 실행 가능
   
    python detect.py --weights yolo5s_fight.onnx --conf 0.4 --source 0

----------
## ref
1. [Python] 지능형 CCTV 만들기, https://velog.io/@supermoony/Python-%EC%A7%80%EB%8A%A5%ED%98%95-CCTV-%EB%A7%8C%EB%93%A4%EA%B8%B0
2. 파이썬으로 email 보내기, https://hyeshin.oopy.io/ds/python/20200620_py20_python_email
3. 라벨링, https://github.com/HumanSignal/labelImg
4. 2024-1. 오픈소스 프로그래밍 강의교안 (Yolov5)
5. 숙명여자대학교 인공지능공학부 김병규 교수님, https://github.com/hopeof-Greatmind/Object-Detection-Yolov5s_custumizedtraining

