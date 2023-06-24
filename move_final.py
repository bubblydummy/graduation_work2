import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
import threading
import socket
import pickle

HOST='192.168.23.29'
PORT=8485

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

client.connect((HOST, PORT))
print("connect")

def my_recv(B_SIZE,client):
    data = client.recv(B_SIZE)
    if not data:
        return data
    cmd = pickle.loads(data)
    return cmd

Motor_A_EN    = 4
Motor_B_EN    = 17

Motor_A_Pin1  = 26
Motor_A_Pin2  = 21
Motor_B_Pin1  = 27
Motor_B_Pin2  = 18

Dir_forward   = 0
Dir_backward  = 1

left_forward  = 0
left_backward = 1

right_forward = 0
right_backward= 1

pwn_A = 0
pwm_B = 0

LED = 11
SWICH = 15

def motorStop():#멈춤
   GPIO.output(Motor_A_Pin1, GPIO.HIGH)
   GPIO.output(Motor_A_Pin2, GPIO.HIGH)
   GPIO.output(Motor_B_Pin1, GPIO.HIGH)
   GPIO.output(Motor_B_Pin2, GPIO.HIGH)
   GPIO.output(Motor_A_EN, GPIO.HIGH)
   GPIO.output(Motor_B_EN, GPIO.HIGH)

def setup():#초기화
   global pwm_A, pwm_B
   GPIO.setwarnings(False)
   GPIO.setmode(GPIO.BCM)
   GPIO.setup(Motor_A_EN, GPIO.OUT)
   GPIO.setup(Motor_B_EN, GPIO.OUT)
   GPIO.setup(Motor_A_Pin1, GPIO.OUT)
   GPIO.setup(Motor_A_Pin2, GPIO.OUT)
   GPIO.setup(Motor_B_Pin1, GPIO.OUT)
   GPIO.setup(Motor_B_Pin2, GPIO.OUT)
   GPIO.setup(LED, GPIO.OUT)
   GPIO.setup(SWICH, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

   motorStop()
   try:
      pwm_A = GPIO.PWM(Motor_A_EN, 1000)
      pwm_B = GPIO.PWM(Motor_B_EN, 1000)
   except:
      pass

def motor_left(status, direction, speed):#Motor 2 positive and negative rotation
   if status == 0: # stop
      GPIO.output(Motor_B_Pin1, GPIO.LOW)
      GPIO.output(Motor_B_Pin2, GPIO.LOW)
      GPIO.output(Motor_B_EN, GPIO.LOW)
   else:
      if direction == Dir_backward:
         GPIO.output(Motor_B_Pin1, GPIO.HIGH)
         GPIO.output(Motor_B_Pin2, GPIO.LOW)
         pwm_B.start(100)
         pwm_B.ChangeDutyCycle(speed)
      elif direction == Dir_forward:
         GPIO.output(Motor_B_Pin1, GPIO.LOW)
         GPIO.output(Motor_B_Pin2, GPIO.HIGH)
         pwm_B.start(0)
         pwm_B.ChangeDutyCycle(speed)

def motor_right(status, direction, speed):#Motor 1 positive and negative rotation
   if status == 0: # stop
      GPIO.output(Motor_A_Pin1, GPIO.LOW)
      GPIO.output(Motor_A_Pin2, GPIO.LOW)
      GPIO.output(Motor_A_EN, GPIO.LOW)
   else:
      if direction == Dir_forward:
         GPIO.output(Motor_A_Pin1, GPIO.HIGH)
         GPIO.output(Motor_A_Pin2, GPIO.LOW)
         pwm_A.start(100)
         pwm_A.ChangeDutyCycle(speed)
      elif direction == Dir_backward:
         GPIO.output(Motor_A_Pin1, GPIO.LOW)
         GPIO.output(Motor_A_Pin2, GPIO.HIGH)
         pwm_A.start(0)
         pwm_A.ChangeDutyCycle(speed)
   return direction


def move(speed, direction, turn, radius=0.6):   # 0 < radius <= 1
    if direction == 'forward':
        if turn == 'left': #좌회전
            motor_left(1, 1, speed)
            motor_right(1, 1, speed)
            time.sleep(0.3)
            
        elif turn == 'right': #우회전
            motor_left(1, 0, speed)
            motor_right(1, 0, speed)
            time.sleep(0.3)
           
        else: #멈춤
            time.sleep(0.3)
            print("stop")
            motorStop()
            time.sleep(1)


def destroy():
   motorStop()
   GPIO.cleanup() # Release resource

def motor1():
    speed=60
    GPIO.output(Motor_A_Pin1, GPIO.LOW)
    GPIO.output(Motor_A_Pin2, GPIO.LOW)
    GPIO.output(Motor_B_Pin1, GPIO.LOW)
    GPIO.output(Motor_B_Pin2, GPIO.LOW)
    GPIO.output(Motor_A_EN, GPIO.LOW)
    GPIO.output(Motor_B_EN, GPIO.LOW)
    while True:
        temp=my_recv(1024, client)
        time.sleep(0.1)
        
        #모터 움직이기
        if temp == "no":
            print("종료")
            break
        else:
            if temp == "forward": speed = 0
            else: speed = 65
            move(speed,'forward',temp)
            print(temp)
            
def camera1():
    while True:
        ret, frame = cam.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)

        data = np.array(frame)
        stringData = data.tobytes()

        client.sendall((str(len(stringData))).encode().ljust(16) + stringData)

if __name__ == '__main__':
    try:
        setup()
        
        #---- 카메라
        cam = cv2.VideoCapture(0)
        cam.set(9, 320)
        cam.set(12, 240)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        #---- 스레드
        receiver = threading.Thread(target=motor1)
        sender = threading.Thread(target=camera1)

        receiver.start()
        sender.start()

        receiver.join()
        sender.join()

        motorStop()
        destroy()
   
    except KeyboardInterrupt:
      destroy()


    #직진 코드
    #      motor_right(1, 0, speed)
    #      motor_left(1, 0, speed)
    #      motor_right(0, 1, speed*0.1)
  