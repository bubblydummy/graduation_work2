import socket
import move_final

SERVER_IP = "0.0.0.0"  # 모든 IP 주소에서 연결을 받음
SERVER_PORT = 10123 #static port

def appCommand(message):
    move_final.move(100, 'forward', message, 0.6)
    
    
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(1)  # 최대 동시 접속 가능한 클라이언트 수

    print("server is started.")

    while True:
        client_socket, client_address = server_socket.accept() #휴대폰 ip 주소, port는 임의 주소
        print("connected", client_address)

        message = client_socket.recv(1024).decode("utf-8")
        print("received message:", message)
        appCommand(message)
        

        client_socket.close()
        
        
        


if __name__ == "__main__":
    main()



