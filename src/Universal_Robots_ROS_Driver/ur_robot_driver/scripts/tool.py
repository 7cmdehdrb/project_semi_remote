#!/usr/bin/env python

import serial

# 시리얼 포트 설정
ser = serial.Serial(
    port='/tmp/ttyUR',  # 이 부분은 실제 시리얼 디바이스에 맞게 변경해야 합니다.
    baudrate=115200,      # 보드레이트 115200으로 설정
    bytesize=serial.EIGHTBITS,  # 데이터 비트는 8비트
    parity=serial.PARITY_EVEN,  # 패리티는 짝수(EVEN)
    stopbits=serial.STOPBITS_ONE,  # 스톱 비트는 1
    timeout=1                     # 타임아웃 설정(필요에 따라 조절 가능)
)

if ser.isOpen():
    ser.close()
ser.open()

# 슬레이브 주소 설정
slave_address = 0x41

# 모드버스 프로토콜을 사용하는 경우 추가적으로 pymodbus와 같은 라이브러리를 사용할 수 있습니다.
# 이 예제에서는 단순히 시리얼 포트를 설정하고 열어 데이터를 송수신하는 방법을 보여줍니다.

# 데이터 송신 예시
ser.write(b'Hello RS485')

# 데이터 수신 예시
while True:
    if ser.inWaiting() > 0:
        received_data = ser.read(ser.inWaiting())  # 대기중인 모든 데이터 읽기
        print(received_data.decode('utf-8'))  # 받은 데이터를 UTF-8로 디코드하여 출력
