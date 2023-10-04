import cv2 # opencv 사용
import numpy as np
import socket
import pickle
import time
def Houghline_array(img): # ------------------------------[ 직선 검출 ]
    min_L = 10 # 선의 최소 길이-->점선
    max_G = 30 # 선사이의 최대 허용간격
    # cv2.HoughLinesP(추출 이미지, 거리정밀도, 세타정밀도, 스레솔드, 선의 최소길이, 선사이의 최대 허용간격)
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 30, minLineLength=min_L, maxLineGap=max_G) # [(시작점), (끝점)] 반환
    
    Line_array=[]
    for line in lines:
        x1 = int(line[0][0]); y1 = int(line[0][1])
        x2 = int(line[0][2]); y2 = int(line[0][3])
        slope = 0
        if (x2-x1) != 0: slope = (y2-y1)/(x2-x1) # 기울기값

        # if abs(slope) > 0.15 and abs(slope) < 0.8: # 기울기값 지정
        if slope != 0:
            x0 = int((height+(slope*x1-y1))/slope) # x절편 계산 y1 - sx1 = x0 , x3 = (y3 + sx1 - y1)/s

            if x0 not in Line_array: # 중복 제거
                Line_array.append([x0,x1,y1,x2,y2,slope])
                # cv2.line(image, (x1, y1), (x2, y2), (50, 50, 255), 2)

    return Line_array

def lane_array(Line_array): # ----------------------------[ 차선 검출 - 모여있는 직선들끼리 묶음 ]
    Line_array.sort() # x절편 오름차순
    Larray=[]; Rarray=[] # 왼쪽 차선, 오른쪽 차선
    set_x0 = []; set_s = []
    for line in Line_array:
        x0 = line[0]
        slope = line[5]
        check = False
        index = Line_array.index(line)

        for j in range(0,len(set_x0)): # 이미 집합에 있는 값과 새로운 값과 비교
            # if abs(set_x0[j] - x0) < 30 and abs(set_s[j] - slope) < 0.2: # 모여있는 정도, 기울기 비슷한 정도--★
            set_x0.append(x0)
            set_s.append(slope) 
            check = True # 저장 확인
            break # 저장했으면 탈출
        
        if check == False or len(Line_array)==index+1: # 집합에 저장이 안됐거나 마지막일 때
            if len(set_x0) > 1:
                # if set_x0[-1]-set_x0[0] >= 5: # 차선 너비--★
                if sum(set_x0)/len(set_x0) < width/2: Larray.append(set_x0) # 왼쪽 차선 저장
                else: Rarray.append(set_x0) # 오른쪽 차선 저장
            if len(Line_array)!=index:
                set_x0 = []; set_s = [] # 집합 초기화
                set_x0.append(x0)
                set_s.append(slope)

    return Larray, Rarray

accuracy=[]

def find_point(image): #----------------------------------[ 곡선 차선의 양끝 위치값 ]
    histo = np.sum(image[image.shape[0]//2:,:],axis=0) # y축을 기준으로 모두 더해 x축만 남김
    histo = [0 if i<=1000 else 1 for i in histo] # 이진화 (1000이하면 0, 아니면 1)
    histo = np.array(histo)
    width = image.shape[1] # 너비
    mid = int(width/2) # x축 중심
    
    histo_rigth_inverse = np.flip(histo[mid:]) # 오른쪽 부분 리스트를 뒤집기
    right = width - np.argmax(histo_rigth_inverse) # 오른쪽 차선의 가장 오른쪽 위치값
    left = np.argmax(histo[:mid]) # 왼쪽 차선의 가장 왼쪽 위치값
    
    return left, right

def sliding_window_search(img):
    nwindows = 5 # 조사창의 개수
    margin = 140 # 조사창의 너비/2
    minpix = 10
    window_height = int(img.shape[0]/nwindows) # 조사창의 높이

    left, right = find_point(img) # 차선의 양끝 위치값 (x값)

    nz = img.nonzero() # 차선이 있는 인덱스 (행(y),열(x))
    
    left_img_inds = []; right_img_inds = []
    Larray, Rarray = [], []
    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((img, img, img))*255 # rgb표현가능하게 3차원으로 쌓기

    for window in range(nwindows):

        win_yh = img.shape[0] - window * window_height # 조사창의 상단 (y값)
        win_yl = img.shape[0] - (window+1) * window_height # 조사창의 하단 (y값)
        
        win_xll = left - margin # 왼쪽 조사창의 왼쪽 (x값)
        win_xlh = left + margin # 왼쪽 조사창의 오른쪽 (x값)
        win_xrl = right - margin # 오른쪽 조사창의 왼쪽 (x값)
        win_xrh = right + margin # 오른쪽 조사창의 오른쪽 (x값)

        # 조사창 그리기
        cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)

        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0] # 왼쪽 조사창에 있는 차선 인덱스
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0] # 오른쪽 조사창에 있는 차선 인덱스

        left_img_inds.append(good_left_inds) # 왼쪽 조사창에 있는 차선들을 조사창의 개수만큼 새로운 배열(left_img_inds)에 저장
        right_img_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix: # 왼쪽 조사창에 있는것이 특정개수를 넘어 차선이라 판단되면
            left = int(np.mean(nz[1][good_left_inds])) # 평균을 구해 새로운 왼쪽 기준점으로 두기
        if len(good_right_inds) > minpix:      
            right = int(np.mean(nz[1][good_right_inds]))

        if window == 0:
            L_x0 = left
            R_x0 = right

        lx.append(left) # 왼쪽조사창의 x값 차선들의 평균을 조사창의 개수만큼 새로운 배열(lx)에 저장
        ly.append((win_yl + win_yh)/2) # 왼쪽조사창의 y값 차선들의 평균을 조사창의 개수만큼 새로운 배열(ly)에 저장
        rx.append(right)
        ry.append((win_yl + win_yh)/2)

    left_img_inds = np.concatenate(left_img_inds) # 1차원 배열로 만들어준다. 
    right_img_inds = np.concatenate(right_img_inds)
    
    out_img[nz[0][left_img_inds], nz[1][left_img_inds]] = [255, 0, 0] # 왼쪽 차선을 빨간색으로
    out_img[nz[0][right_img_inds] , nz[1][right_img_inds]] = [0, 0, 255] # 오른쪽 차선을 파란색으로

    cv2.imshow("viewer", out_img)
    
    # polyfit 함수로 ly, lx 의 점들로 그린 함수의 계수값을 찾아준다. 2차로 polyfit 하므로 ax^2+bx+c 에서 a,b,c 값을 가져옴
    lfit = np.polyfit(np.array(ly),np.array(lx),2)
    rfit = np.polyfit(np.array(ry),np.array(rx),2)

    return lfit, rfit, L_x0, R_x0
   
def draw_lane(image, warp_img, mat, left_fit, right_fit): #---------------[ 원근 역변환을 통해 원래 이미지에 그래프를 그림 ]
    
    yMax = warp_img.shape[0]#540
    ploty = np.linspace(0, yMax - 1, yMax)#0~yMax-1까지 yMax개수만큼 채우기
    
    warp_zero = np.zeros_like(warp_img).astype(np.uint8)#warp_img 의 크기만큼 0으로 채운 ndarray 반환
    color_warp=np.dstack((warp_zero,warp_zero,warp_zero))#RGB 3차원 이미지로 변경
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] #ax^2+bx+c값으로 선형회귀
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))#왼쪽 차선(양끝)과 오른쪽 차선(양끝) 합쳐서 배열로저장

    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))#RGB 이미지에 pts크기의 도형 그리기
    #cv2.imshow('hello',color_warp)
    newwarp = cv2.warpPerspective(color_warp, mat, (image.shape[1], image.shape[0]))#원근 역변환하여 color_warp 을 적용
    #color_warp에서 mat(원근 역변환한 크기) 만큼을 width,heigth크기로 만든다.

    result=cv2.addWeighted(image, 1, newwarp, 0.3, 0) # 원본이미지에 그린 도형 합치기
    
    return result

def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def my_send(cmd, client): # 데이터 보내기
    data = pickle.dumps(cmd)
    client.sendall(data)

HOST='192.168.107.29' #pc ip
PORT=8485 #pc port

# 1. 초기화
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')


# 2. bind
server.bind((HOST,PORT))
print('Socket bind complete')

# 3. listen
server.listen(1)
print('Socket now listening')

# 4. accept
client,addr = server.accept()

i = 0; num1 = 0; num2 = 0; c_r=0; c_l=0; before_move="no"
#-------<라즈베리 파이 웹캠에서 받아오기>-------
while 1:
    # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
    length = recvall(client, 16)
    stringData = recvall(client, int(length))
    data = np.fromstring(stringData, dtype = 'uint8')
    #data를 디코딩한다.
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    #------------< 특정 색 추출 >------------
    # BGR 흰색
    bgrLower = np.array([220, 220, 220]) # 하한(BGR)
    bgrUpper = np.array([255, 255, 255]) # 상한(BGR)
    bgr_mask = cv2.inRange(image, bgrLower, bgrUpper) # BGR에서 흰색 픽셀 추출
    white_img = cv2.bitwise_and(image, image, mask=bgr_mask) # 원본 이미지와 마스크를 합성
    # HSV 노란색
    # hsvLower = np.array([18, 70, 100]) # 하한(HSV)
    # hsvUpper = np.array([24, 255, 255]) # 상한(HSV)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환
    # hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper) # HSV에서 노란색 픽셀 추출
    # yellow_img = cv2.bitwise_and(image, image, mask=hsv_mask) # 원래 이미지와 마스크를 합성

    # WnY_img = cv2.bitwise_or(white_img, yellow_img) # 흰색&노란색 이미지 합성

    #-------------< 윤곽선 추출 >-------------
    gray_img = cv2.cvtColor(white_img, cv2.COLOR_RGB2GRAY) # 흑백이미지로 변환    
    ga_img=cv2.GaussianBlur(gray_img, (3, 3), 0) # 3*3 가우시안 필터 적용
    canny_img = cv2.Canny(ga_img,100,200) # Canny edge 알고리즘
    sobel_img = cv2.Sobel(ga_img, cv2.CV_8U, 1, 0, 3)

    #------------< 관심 영역 지정 >-----------
    height = image.shape[0]
    width = image.shape[1]
    white_color = (255,255,255)
    point = np.array([[0,height],[0,height/2],[width,height/2],[width,height]], np.int32) # 관심 영역 좌표
    
    black_img = np.zeros_like(sobel_img) # 검은색 배경
    fill_img = cv2.fillPoly(black_img, [point], white_color) # 관심 영역
    edges_img = cv2.bitwise_and(sobel_img, fill_img) # 관심 영역 안의 윤곽선 추출

    # ---< 이미지 변환을 위한 관심 영역 지정 >---
    p1=[0,height//2+20] # 좌상
    p2=[width,height//2+20] # 우상
    p3=[0,height-10] # 좌하
    p4=[width,height-10] # 우하
    
    #--------------< 위치 확인 >--------------
    blue_color=(255,0,0)
    # cv2.circle(image,p1,5,blue_color,-1)
    # cv2.circle(image,p2,5,blue_color,-1)
    # cv2.circle(image,p3,5,blue_color,-1)
    # cv2.circle(image,p4,5,blue_color,-1)
    
    #-------------< 이미지 변환 >-------------
    src = np.float32([p1,p2,p3,p4])
    dst = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv2.getPerspectiveTransform(src, dst) # 원근 변환(관심영역, 원본영역)
    matrix_inv = cv2.getPerspectiveTransform(dst, src) # 원근 역변환
    trans = cv2.warpPerspective(edges_img, matrix, (width,height)) #(적용될 이미지, 원근변환으로 만들어진 행렬, 화면크기)

    #--------------< 곡선 검출 >--------------
    lfit, rfit, L, R = sliding_window_search(trans)
    c_L = p3[0] + L*(p4[0]-p3[0])/width
    c_R = p3[0] + R*(p4[0]-p3[0])/width
    c_mid = (c_L + c_R)/2 # 차선 중심

    lane_img = image.copy()
    lane_img = draw_lane(image, trans, matrix_inv, lfit, rfit) # 그리기
    
    # #--------------< 직선 검출 >----------- q---
    # Line_array = Houghline_array(edges_img)
    
    # #--------------< 차선 검출 >--------------
    # Larray, Rarray = lane_array(Line_array)

    # for line in Line_array:
    #     x0 = line[0]
    #     x1 = line[1]; y1 = line[2]
    #     x2 = line[3]; y2 = line[4]
    #     slope = line[5]

    #     if len(Larray) != 0 and len(Rarray) != 0:
    #         if x0 in Larray[-1]: cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # 왼쪽 차선
    #         elif x0 in Rarray[0]: cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # 오른쪽 차선
    #         else: cv2.line(lane_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            

    # #------------< 차선 치우침 판단 >------------
    # if len(Larray) != 0 and len(Rarray) != 0:
    #     L_x0 = int(sum(Larray[-1])/len(Larray[-1])) # 왼쪽 차선 x절편
    #     R_x0 = int(sum(Rarray[0])/len(Rarray[0])) # 오른쪽 차선 x절편
    #     m_x0 = (L_x0 + R_x0)/2 # 차선 중심
        
    #     move = "no"
    #     if (R_x0-L_x0) < width and (R_x0-L_x0) > width/2: # 차선 간의 간격--★
    #         if m_x0 < width/2-10: move = "left" # 차는 오른쪽으로
    #         elif m_x0 > width/2+10: move = "right" # 차는 왼쪽으로
    #         # print("직선-", int(L_x0), int(R_x0), move)
    #         #print("직선-",move)
    #         #my_send(move,client)
    #         num1 += 1

    #     # 직선
    #     cv2.circle(lane_img,(int(L_x0),int(height-5)),7,(0,0,255),-1)
    #     cv2.circle(lane_img,(int(R_x0),int(height-5)),7,(0,0,255),-1)
    #     cv2.circle(lane_img,(int(m_x0),int(height-5)),7,(0,0,255),-1)

    #else: # 직선 인식 안되면 곡선
    move="no"
    
        
    if c_mid < width/2-50 : move = "right"; c_r+=1 # 왼쪽 바퀴 움직이기
    elif c_mid > width/2+50 : move = "left"; c_l+=1 # 오른쪽 바퀴 움직이기
    
    if before_move=="left" and move=="no":
        c_l=0; c_r=0
        print("갱신")
    if before_move=="right" and move=="no":
        c_r=0; c_l=0
        print("갱신")
    
    #if move=="no": c_l=0; c_r=0
    
 
    if (move=="left" and c_l==1) or (move=="right" and c_r==1):
        print(move)
        my_send(move,client)
        
    before_move=move
        
    
    #my_send(move,client)
    # # 곡선
    cv2.circle(lane_img,(int(c_L),int(p3[1])),7,(0,255,0),-1)
    cv2.circle(lane_img,(int(c_R),int(p3[1])),7,(0,255,0),-1)
    cv2.circle(lane_img,(int(c_mid),int(p3[1])),7,(0,255,0),-1)

    # -------------< 영상 출력 >--------------
    cv2.imshow("sobel_img", sobel_img)
    cv2.imshow("canny_img",canny_img)
    cv2.imshow("result", lane_img)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1

# print("차선 인식률:",num1/i*100,"%") # 차선 이동 방향 인식률
# print("이동 방향 인식률:",num2/num1*100,"%") # 차선 이동 방향 인식률

client.close()
server.close()
print('연결 종료')
cv2.destroyAllWindows()