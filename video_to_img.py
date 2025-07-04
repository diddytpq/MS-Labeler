import cv2
import os

# 입력 비디오 파일 경로
video_path = "./videos/test2.mp4"

# 출력 이미지 저장 경로
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 인덱스 초기화
frame_index = 0
save_index = 0


# 비디오 읽기 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    if frame_index % 10 == 0:  # 10프레임마다 저장
        # 이미지 파일 이름 생성
        frame_filename = f"{output_dir}/{save_index:08d}.jpg"

        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")

        save_index += 1
        
    # 프레임 인덱스 증가
    frame_index += 1

# 비디오 캡처 객체 해제
cap.release()
print("모든 프레임 저장 완료.")
