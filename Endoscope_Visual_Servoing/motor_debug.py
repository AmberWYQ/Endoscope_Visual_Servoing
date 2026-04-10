import serial
import time
import numpy as np

# 串口配置（参考 yoloe_control_main.py）
PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200
MAX_SPEED = 100  # 调试阶段建议使用极低的速度

def send_raw_cmd(ser, m1, m2):
    """直接发送底层命令字符串"""
    # 格式化为 4 个 6 位宽的整数
    cmd_str = f"{m1:>6d}{m2:>6d}{0:>6d}{0:>6d}"
    print(f"Sending: {cmd_str}")
    ser.write(cmd_str.encode('ascii'))

def motor_test():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {PORT}. Use WASD to test, Space to stop, Q to quit.")
        
        while True:
            key = input("Input (w/s/a/d/space/q): ").lower()
            
            if key == 'w':
                # 测试 Motor 1 正向
                send_raw_cmd(ser, MAX_SPEED, 0)
            elif key == 's':
                # 测试 Motor 1 负向
                send_raw_cmd(ser, -MAX_SPEED, 0)
            elif key == 'a':
                # 测试 Motor 2 正向
                send_raw_cmd(ser, 0, MAX_SPEED)
            elif key == 'd':
                # 测试 Motor 2 负向
                send_raw_cmd(ser, 0, -MAX_SPEED)
            elif key == ' ' or key == '':
                # 停止
                send_raw_cmd(ser, 0, 0)
            elif key == 'q':
                send_raw_cmd(ser, 0, 0)
                break
            
            # 持续一小段时间后自动停止，防止失控
            time.sleep(2)
            send_raw_cmd(ser, 0, 0)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    motor_test()