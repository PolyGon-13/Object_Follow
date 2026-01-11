import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os

pt_path='../weights/yolov8n.pt'

def main():
    model=YOLO(pt_path)

    # Realsense 파이프라인 및 설정 객체 생성
    pipeline=rs.pipeline()
    config=rs.config()

    # 스트림 설정
    config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30) # 컬러 스트림 활성화
    # config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30) # 깊이 스트림 활성화 (현재 코드에서는 사용되지 X)

    # bgr8은 색상정보를 파랑,초록,빨강 채널 순서로 각 채널당 8비트(0~255)로 표현하는 이미지 형식
    # bgr8은 opencv에서 사용하는 표준 컬러 포맷이기에 해당 형식으로 데이터를 받으면 바로 처리할 수 있음

    # 카메라 스트리밍 시작
    pipeline.start(config)

    try:
        while True:
            # 프레임 대기 및 수신
            frames=pipeline.wait_for_frames()
            color_frame=frames.get_color_frame() # 컬러 프레임만 추출

            if not color_frame:
                continue
            
            img=np.asanyarray(color_frame.get_data()) # 프레임을 NumPy 배열로 변환 (img 배열이 yolo모델과 opencv에서 처리할 수 있는 형태)

            results=model(img,verbose=False) # yolo 모델로 객체 탐지 수행 (verbose=False는 터미널에 불필요한 로그가 출력되지 않도록 함)

            annotated_frame=results[0].plot() # 탐지 결과를 이미지에 그리기

            cv2.imshow("YOLO_RealSense",annotated_frame) # 결과 프레임 출력

            if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 누르면 루프 종료
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()