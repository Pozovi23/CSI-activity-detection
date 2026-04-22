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
                        help="Window size (number of lines per device per segment)")

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
# GLOBAL SYNCHRONIZED BUFFERS
# ==============================

device_buffers = [[], [], []]   # три буфера для трёх устройств
buffer_lock = threading.Lock()
test_counter = 1
window_size = None
overlap = None
output_dir = None


def try_save_segment():
    global test_counter
    with buffer_lock:
        # Проверяем, что во всех буферах есть хотя бы window_size строк
        if all(len(buf) >= window_size for buf in device_buffers):
            # Создаём новую папку test_N
            folder_name = f"test_{test_counter}"
            folder_path = os.path.join(output_dir, folder_name)
            ensure_dir(folder_path)
            print(f"\n[SYNC] Creating {folder_path}")

            # Для каждого устройства сохраняем первые window_size строк в devX.data
            for idx, buf in enumerate(device_buffers):
                segment = buf[:window_size]
                filename = f"dev{idx+1}.data"
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(HEADER)
                    writer.writerows(segment)
                print(f"  Saved {filename} ({len(segment)} rows)")

                # Обрезаем буфер с учётом overlap
                device_buffers[idx] = buf[window_size - overlap:]

            test_counter += 1


# ==============================
# SERIAL LISTENER
# ==============================

def serial_listener(device_index, port, baudrate):
    print(f"[dev{device_index+1}] Opening {port}...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"[dev{device_index+1}] Cannot open {port}: {e}")
        return

    print(f"[dev{device_index+1}] Connected!")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if "CSI_DATA" in line:
                parsed = parse_csi_line(line)
                if parsed:
                    row = [logger_timestamp()] + parsed
                    with buffer_lock:
                        device_buffers[device_index].append(row)
                    # После добавления проверяем, можно ли сохранить сегмент
                    try_save_segment()
        except Exception as e:
            print(f"[dev{device_index+1}] Error:", e)
            break


# ==============================
# MAIN
# ==============================

def main():
    global window_size, overlap, output_dir

    args = parse_args()

    if args.overlap >= args.window:
        raise ValueError("Overlap must be smaller than window size")

    window_size = args.window
    overlap = args.overlap
    output_dir = args.output

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

    print("\nAll devices detected. Starting synchronized logging...\n")

    threads = []
    for i, port in enumerate(args.ports):
        t = threading.Thread(
            target=serial_listener,
            args=(i, port, args.baud),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping... Saving remaining data in buffers...")
        # Сохраняем остатки, даже если не набрался полный window
        with buffer_lock:
            if any(device_buffers):
                folder_name = f"test_{test_counter}_partial"
                folder_path = os.path.join(output_dir, folder_name)
                ensure_dir(folder_path)
                print(f"[SYNC] Saving partial data to {folder_path}")
                for idx, buf in enumerate(device_buffers):
                    if buf:
                        filename = f"dev{idx+1}.data"
                        filepath = os.path.join(folder_path, filename)
                        with open(filepath, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(HEADER)
                            writer.writerows(buf)
                        print(f"  Saved {filename} ({len(buf)} rows)")
        print("Done.")

if __name__ == "__main__":
    main()