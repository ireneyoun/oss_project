import cv2
import xml.etree.ElementTree as ET
import os

# 비디오 파일을 전처리하는 함수
def preprocess_video(video_path, label_path, output_dir):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    # 프레임 인덱스 초기화
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 라벨링 데이터에서 해당 프레임의 바운딩 박스 추출
        boxes = parse_xml(label_path, frame_idx)
        # 바운딩 박스가 있는 경우 프레임 저장
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                # 바운딩 박스 좌표 추출
                xmin, ymin, xmax, ymax = box
                # 바운딩 박스를 빨간색으로 프레임에 그리기
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # 바운딩 박스 주석 달기
                text = f"Object {i+1}"
                cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 저장된 프레임 이미지 파일 경로
        output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg")
        # 프레임 이미지 저장
        cv2.imwrite(output_path, frame)
        frame_idx += 1
    cap.release()

# XML 파일에서 객체의 바운딩 박스를 파싱하는 함수
def parse_xml(xml_file, frame_idx):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for track in root.findall('track'):
        label = track.get('label')
        if label == "fight_start" and int(track.find('box').get('frame')) == frame_idx:
            box = track.find('box')
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr'))
            boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return boxes

# 비디오 디렉토리 경로
videos_dir = "/Users/zero/Desktop/오픈소스/테스트용/원천데이터/"
# 라벨링 데이터(XML) 디렉토리 경로
labels_dir = "/Users/zero/Desktop/오픈소스/테스트용/라벨링데이터/"
# 전처리된 프레임 이미지를 저장할 디렉토리 경로
output_dir = "/Users/zero/Desktop/오픈소스/테스트용/output/"

# 전처리된 프레임 이미지를 저장할 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 비디오 디렉토리 내의 모든 파일 목록 가져오기
video_files = os.listdir(videos_dir)
for video_file in video_files:
    # MP4 확장자를 가진 파일인지 확인
    if video_file.endswith(".mp4"):
        # 각 비디오 파일의 경로 설정
        video_path = os.path.join(videos_dir, video_file)
        # 비디오 파일에 대한 라벨링 데이터(XML) 파일의 경로 설정
        label_file = os.path.splitext(video_file)[0] + ".xml"
        label_path = os.path.join(labels_dir, label_file)
        # 비디오 파일 전처리 함수 호출
        preprocess_video(video_path, label_path, output_dir)

