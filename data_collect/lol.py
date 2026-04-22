import serial
import serial.tools.list_ports
import threading
import os
import time
import argparse
from datetime import datetime
import csv

# ==============================
# ARGUMENT PARSER
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(description="CSI Logger for 3 ESP32-S3")

    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")

    parser.add_argument("--window", type=int, required=True,
                        help="Window size (number of lines per file)")

    parser.add_argument("--overlap", type=int, required=True,
                        help="Overlap size (number of lines)")

    parser.add_argument("--ports", nargs=3, required=True,
                        help="Three serial ports (e.g., COM5 COM6 COM7)")

    parser.add_argument("--baud", type=int, default=921600,
                        help="Baudrate (default 921600)")

    return parser.parse_args()


# ==============================
# UTILS
# ==============================

HEADER = [
    "logger_timestamp",
    "record_type",
    "packet_seq",
    "source_mac",
    "rssi_dbm",
    "rate_code",
    "sig_mode",
    "mcs_index",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor_dbm",
    "ampdu_count",
    "wifi_channel",
    "secondary_channel",
    "local_timestamp",
    "antenna",
    "signal_length",
    "rx_state",
    "csi_len",
    "first_word",
    "csi_data"
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def logger_timestamp():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


# ==============================
# CHECK CONNECTIONS
# ==============================

def check_all_connected(required_ports):
    available = list_available_ports()
    print("\nAvailable ports:", available)

    all_ok = True

    for port in required_ports:
        if port not in available:
            print(f"[ERROR] {port} not found!")
            all_ok = False
        else:
            print(f"[OK] {port} detected")

    return all_ok


# ==============================
# PARSE CSI LINE
# ==============================

def parse_csi_line(raw_line):

    try:
        parts = raw_line.split(",", 24)

        if len(parts) < 25:
            return None

        return parts

    except:
        return None


# ==============================
# SERIAL LISTENER
# ==============================

def serial_listener(rx_name, port, baudrate, output_dir, window_size, overlap):

    rx_folder = os.path.join(output_dir, rx_name)
    ensure_dir(rx_folder)

    buffer = []
    file_index = 0

    print(f"[{rx_name}] Opening {port}...")

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"[{rx_name}] Cannot open {port}: {e}")
        return

    print(f"[{rx_name}] Connected!")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()

            if "CSI_DATA" in line:

                parsed = parse_csi_line(line)

                if parsed:
                    row = [logger_timestamp()] + parsed
                    buffer.append(row)

                if len(buffer) >= window_size:
                    save_segment(rx_folder, buffer, file_index)
                    file_index += 1
                    buffer = buffer[window_size - overlap:]

        except Exception as e:
            print(f"[{rx_name}] Error:", e)
            break


def save_segment(folder, buffer, index):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{index}.csv"
    path = os.path.join(folder, filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(buffer)

    print(f"[SAVED] {path}")


# ==============================
# MAIN
# ==============================

def main():
    args = parse_args()

    if args.overlap >= args.window:
        raise ValueError("Overlap must be smaller than window size")

    print("\n========= CONFIGURATION =========")
    print("Output directory:", args.output)
    print("Window size:", args.window)
    print("Overlap:", args.overlap)
    print("Ports:", args.ports)
    print("Baudrate:", args.baud)
    print("=================================\n")

    ensure_dir(args.output)

    print("Checking ESP32 connections...")

    while not check_all_connected(args.ports):
        print("Waiting for all devices...")
        time.sleep(3)

    print("\nAll devices detected. Starting logging...\n")

    threads = []

    for i, port in enumerate(args.ports):
        rx_name = f"RX{i+1}"

        t = threading.Thread(
            target=serial_listener,
            args=(rx_name, port, args.baud,
                  args.output, args.window, args.overlap),
            daemon=True
        )

        t.start()
        threads.append(t)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
