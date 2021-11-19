import threading
import csv
import serial
from datetime import datetime
import struct
import numpy as np
import pandas as pd
from concurrent import futures
import time


class SerialPort:
    def __init__(self, port, buand):
        self.port = serial.Serial(port, buand)
        self.port.close()
        if not self.port.isOpen():
            self.port.open()

    def port_open(self):
        if not self.port.isOpen():
            self.port.open()

    def port_close(self):
        self.port.close()

    def send_data(self):
        self.port.write('')

    def read_data(self):
        global is_exit
        global data_bytes
        # self.port.set_buffer_size(2000)
        while not is_exit:

            # if data_csv.shape[1] > 100:
            #     is_exit = True

            count = self.port.inWaiting()
            if count > 0:
                rec_str = self.port.read(count)
                data_bytes += rec_str
                # print(len(data_bytes))
                # print(data_csv)
                # print(data_bytes)
                # print('当前数据接收总字节数：'+str(len(data_bytes))+' 本次接收字节数：'+str(len(rec_str)))
                # print(str(datetime.now()),':',binascii.b2a_hex(rec_str))
        return 'Reading finished!!!'

    # ___________数据解析____________

    def parser(self):
        global is_exit
        global data_bytes

        while not is_exit:

            # if data_csv.shape[1] > 100:
            #     is_exit = True

            data_len = len(data_bytes)
            # print(data_len)
            i = 0

            while i < data_len - 1:
                try:
                    # print(data_bytes)
                    if (data_bytes[i:i + 1][0] == 0x55 and data_bytes[i + 1:i + 2][0] == 0x55
                            and data_bytes[i + 2:i + 3][0] == 0x1):

                        try:
                            data_frame = data_bytes[i + 4:i + 10]

                            Roll, Pitch, Yaw = struct.unpack('<hhh', data_frame)
                            dt = str(datetime.now()).split()[1][:-7]

                        except Exception as e:
                            # print(e)
                            pass
                        else:

                            Roll = float(Roll) / 32768 * 180
                            Pitch = float(Pitch) / 32768 * 180
                            Yaw = float(Yaw) / 32768 * 180

                            print(dt, Roll, Pitch, Yaw)

                            with open(r'.\database\serial.csv','a',newline='') as csvff:
                                csvw=csv.writer(csvff)
                                csvw.writerow([dt,Roll,Pitch,Yaw])

                            i += 11

                        # print(data_csv)

                    else:
                        i += 1
                        # print(data_csv)
                        # print('没有接收到数据')


                except IndexError as e:
                    # print(e)
                    pass

            data_bytes[:i] = b''


if __name__ == '__main__':

    # ___________设定串口____________

    serialPort = 'COM3'  # 串口
    baudRate = 115200  # 波特率
    is_exit = False
    data_bytes = bytearray()

    # ___________开启数据读取____________
    # 后期可以在此处扩展opencv视觉感知模块，同步两者数据。

    mySerial = SerialPort(serialPort, baudRate)

    with futures.ThreadPoolExecutor(2) as executor:

        result_of_reading = executor.submit(mySerial.read_data)

        result_of_parser = executor.submit(mySerial.parser())

        for result in futures.as_completed([result_of_reading, result_of_parser]):
            print(result.result())
