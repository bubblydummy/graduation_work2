## 차선 인식 및 제어

### 1. 목적 및 필요성

- 목적 : 차선을 정확히 인지하여 주행 중 방향지시등이 꺼져 있을 때, 차선 변경을 시도시, 차선 변경을 제한하여 다시 원래 차선으로 돌아오게 한다.
- 필요성 : 방향지시등을 켜지 않고 차선을 이동하여 발생하는 사고를 막기 위해서이다.


### 2. 기능블록도 및 기능 순서도
![기능 블록도](https://drive.google.com/file/d/1nRxMBZr2pE70MxwfhrSsyuiaAnNc4OA8/view?usp=drive_link)

![기능 순서도](https://drive.google.com/file/d/1EniO4pN1vLiTWE5ljwEXwl27Bytuj9_i/view?usp=drive_link)
### 3. 개발 내용
![직선 차선 인식 알고리즘](https://drive.google.com/file/d/17zjvMJy6FdrFSHdUVU6uKsuTUkz5NWS8/view?usp=drive_link)

![차선 변경 판단 코드](https://drive.google.com/file/d/1DiZsSwUUL-XvM-56k-il5hizDTkGIfte/view?usp=drive_link)

![차선 변경 판단 코드 부연 설명](https://drive.google.com/file/d/1Vc3X3a7-mYaHcEeiG1E7cxc9oTDQcgyT/view?usp=drive_link)
### 4. 기대 효과
- 방향지시등이 꺼져 있을 시, 차선 이탈 방지
- 곡선 및 직선 차선 인식
- 곡선 차선의 한계로 차선 내 흰색 장애물이 있을 시, 차선을 정확히 인지할 수 없는데, 이를 직선 차선 알고리즘을 이용하여 보완한다.

### 5. 작품 사진
![작품 사진](https://drive.google.com/file/d/1vInYgFhpTso8YYJvabP_MLqvdTmX6yzW/view?usp=drive_link)
### 6. 보완 및 수정이 필요한 내용
- 영상처리 속도 향상
- LED 제어 미구현
### 7. 코드와 영상 파일 관계도

- merge_final <-> game.mp4

- merge_final <-> line.mp4(곡선 및 직선 차선)

- merge_final <-> hough_line.mp4(직선차선)

- 1line_socket_final.py(사용자 인터 페이스) <-> move_final.py(라즈베리파이 내부)

merge_final.py,line_socket_final.py,game.mp4,line.mp4,hough_line.mp4 모두 동일 디렉토리에 위치
move_final.py 라즈베리파이 내부에 위치

  
