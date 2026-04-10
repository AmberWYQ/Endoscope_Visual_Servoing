"""
Serial Manager for Visual Servoing Control

Based on the original serial_manager.py but adapted for inference control loop.
Handles communication with the robot's motor controllers.
"""

import serial
import serial.tools.list_ports
import time
import threading


class SerialManager:
    """Manages serial communication with the robot's motor controllers."""
    
    def __init__(self, baud_rate=115200, auto_connect=True):
        """
        Initialize the serial manager.
        
        Args:
            baud_rate: Serial communication baud rate
            auto_connect: If True, interactively select and connect to port
        """
        self.baud = baud_rate
        self.ser = None
        self.lock = threading.Lock()
        self.last_send_time = 0
        self.send_interval = 0.02  # 50Hz max send rate
        self.port = None
        self.connected = False
        
        if auto_connect:
            self.port = self._interactive_select_port()
            if self.port:
                self.connect()

    def _interactive_select_port(self):
        """
        List all serial ports and allow user to select one interactively.
        
        Returns:
            Selected port device path, or None if no ports available
        """
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("\n[Serial] ❌ No serial devices detected! Check USB connection.")
            return None
        
        print("\n" + "=" * 50)
        print("[Serial] Available serial devices:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p.device}  <-- {p.description}")
        print("=" * 50)

        while True:
            user_input = input(f"👉 Select port (0-{len(ports)-1}) [default 0]: ").strip()
            
            if user_input == "":
                selected_index = 0
                break
            
            if user_input.isdigit():
                idx = int(user_input)
                if 0 <= idx < len(ports):
                    selected_index = idx
                    break
            
            print(f"⚠️ Invalid input: '{user_input}', please enter a valid number.")

        selected_device = ports[selected_index].device
        print(f"✅ Selected port: {selected_device}\n")
        return selected_device

    def connect(self):
        """
        Establish serial connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.port:
            return False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.connected = True
            print(f"[Serial] Connected to {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"[Serial] Connection failed: {e}")
            self.connected = False
            return False

    def reconnect(self):
        """Attempt to reconnect after connection loss."""
        time.sleep(1.0)
        print(f"[Serial] Attempting reconnection...")
        self.connect()

    def write_raw(self, data):
        """
        Write raw data to serial port.
        
        Args:
            data: Bytes to send
        """
        if not self.ser or not self.ser.is_open:
            self.reconnect()
            return False
            
        try:
            with self.lock:
                self.ser.write(data)
                self.last_send_time = time.time()
            return True
        except Exception as e:
            print(f"[Serial] Send error: {e}")
            self.reconnect()
            return False

    def send_motor_packet(self, speeds):
        """
        Send motor speed commands.
        
        Args:
            speeds: List of motor speeds [M1, M2] (or [M1, M2, M3, M4] for 4-motor systems)
        
        The protocol expects 4 motor values, so we pad with zeros if needed.
        """
        # Ensure we have at least 2 values
        while len(speeds) < 2:
            speeds.append(0)
        
        # Pad to 4 values for the protocol
        while len(speeds) < 4:
            speeds.append(0)
        
        # Clamp to valid range (-10000 to 10000)
        clamped = [max(min(int(s), 10000), -10000) for s in speeds]
        
        # Format as fixed-width string
        cmd_str = "{:>6d}{:>6d}{:>6d}{:>6d}".format(
            clamped[0], clamped[1], clamped[2], clamped[3]
        )
        return self.write_raw(cmd_str.encode('ascii'))

    def send_stop(self):
        """Send stop command (all motors to zero)."""
        return self.send_motor_packet([0, 0, 0, 0])

    def close(self):
        """Close the serial connection."""
        if self.ser:
            self.send_stop()  # Stop motors before closing
            time.sleep(0.05)
            self.ser.close()
            self.connected = False
            print("[Serial] Connection closed")


class MockSerialManager:
    """
    Mock serial manager for simulation mode (no real robot).
    Provides the same interface but doesn't send any actual commands.
    """
    
    def __init__(self, baud_rate=115200):
        self.baud = baud_rate
        self.port = "MOCK"
        self.connected = True
        self.last_command = [0, 0]
        print("[Serial] Running in SIMULATION mode (no real robot)")
    
    def connect(self):
        return True
    
    def reconnect(self):
        pass
    
    def write_raw(self, data):
        return True
    
    def send_motor_packet(self, speeds):
        self.last_command = speeds[:2] if len(speeds) >= 2 else [0, 0]
        return True
    
    def send_stop(self):
        self.last_command = [0, 0]
        return True
    
    def close(self):
        pass
