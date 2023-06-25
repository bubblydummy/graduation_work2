## 차선 인식 및 제어

### 1. 목적 및 필요성

- 목적 : 차선을 정확히 인지하여 주행 중 방향지시등이 꺼져 있을 때, 차선 변경을 시도시, 차선 변경을 제한하여 다시 원래 차선으로 돌아오게 한다.
- 필요성 : 방향지시등을 켜지 않고 차선을 이동하여 발생하는 사고를 막기 위해서이다.


### 2. 기능블록도 및 기능 순서도
기능 블록도 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/4#issue-1773328722.png"/>

기능 순서도 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/2#issue-1773327698.png"/>

### 3. 개발 내용
직선 차선 인식 알고리즘 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/1#issue-1773325152.png"/>

차선 변경 판단 코드 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/5#issue-1773328829.png"/>


차선 변경 판단 코드 부연 설명 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/6#issue-1773328952.jpg"/>

### 4. 기대 효과
- 방향지시등이 꺼져 있을 시, 차선 이탈 방지
- 곡선 및 직선 차선 인식
- 곡선 차선의 한계로 차선 내 흰색 장애물이 있을 시, 차선을 정확히 인지할 수 없는데, 이를 직선 차선 알고리즘을 이용하여 보완한다.

### 5. 작품 사진
작품 사진 : 
<img width="80%" src="https://github.com/bubblydummy/graduation_work2/issues/3#issue-1773328294"/>

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

  
