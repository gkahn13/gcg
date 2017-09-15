import serial
import glob
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


class SensorsHandler:
    def __init__(self, p=50.0, d=0.0, timeout=0.1, is_plotting=False):
        self._motor_ser = None
        self._imu_ser = None
        self._motor_data = None
        self._last_motor_data = None  # used for pid
        self._imu_data = None  # (pos, hpr)
        self._motor_cmd = None  # (steer, motor)
        self._last_motor_cmd = None
        self._max_vel = 0
        self._vel_cmd = None  # (steer, vel)
        self._is_calibrated = False
        self._default_defined = False
        self._default_steer_motor = None
        self._default_imu_data = None
        self._crashed = False
        self._flip = False
        self._new_vel_cmd = False

        self._cam = None
        self._image = None
        self._threads = []
        # For pid controller
        self._p = p
        self._d = d
        self._last_err = 0
        self._dt = 0

        # For graphing
        self._is_plotting = is_plotting
        if self._is_plotting:
            self._times = []
            self._steer_values = []
            self._motor_values = []
            self._encoder_values = []

            self._x_values = []
            self._y_values = []
            self._z_values = []

            self._setup_plot()

        serial_ports = glob.glob('/dev/ttyACM*')
        sers = [serial.Serial(port, 115200, timeout=timeout) for port in serial_ports]
        print('Found {0} Serial Ports'.format(len(sers)))
        for ser in sers:
            data = self._read_ser(ser)
            while len(data) == 0 or data[0] != '(' or data[-3] != ')':
                data = self._read_ser(ser)
            if data[:4] == '(imu':
                self._imu_ser = ser
                print('IMU found')
                self._threads.append(threading.Thread(target=self._imu_thread))
            else:
                self._motor_ser = ser
                print('Motor found')
                self._threads.append(threading.Thread(target=self._motor_thread))

        for cam_num in [0, 1]:
            cam = cv2.VideoCapture(cam_num)
            if cam.read()[0]:
                self._cam = cam
                self._cam.set(3, 160)
                self._cam.set(4, 90)
                self._cam.set(11, 50)
                self._cam.set(13, 50)
                print('Camera found')
                self._threads.append(threading.Thread(target=self._video_thread))
                break
        assert (self._imu_ser is not None)
        assert (self._motor_ser is not None)
        assert (self._cam is not None)
        print('Starting Threads')
        #        self._threads.append(threading.Thread(target=self._motor_thread))
        #        self._threads.append(threading.Thread(target=self._imu_thread))
        #        self._threads.append(threading.Thread(target=self._video_thread))
        for t in self._threads:
            t.daemon = True
            t.start()
        self.calibrate()

    # Threading

    def _motor_thread(self):
        #        try:
        while self._motor_ser.is_open:
            data = self._read_ser(self._motor_ser)
            if len(data) > 0 and data[0] == '(' and data[-3] == ')':
                motor_data = np.array(data[1:-3].split(","), dtype=np.float32)
                if len(motor_data) == 4:
                    if self._default_defined:
                        steer_diff, motor_diff = motor_data[1:3] - self._default_steer_motor
                        if motor_diff == 0:
                            motor_sign = 0
                        else:
                            motor_sign = abs(motor_diff) / motor_diff
                        self._motor_data = [motor_data[0], steer_diff, motor_diff, motor_data[3] * motor_sign]
                    else:
                        self._default_steer_motor = motor_data[1:3]

            if self._is_calibrated:
                self._max_vel = max(self._max_vel, self._motor_data[3])

                if self._crashed or self._flip:
                    self._motor_cmd = (0.0, 0.0)
                    self._vel_cmd = None
                elif self._vel_cmd is not None and (self._new_vel_cmd or self._motor_data is not self._last_motor_data):
                    self._new_vel_cmd = False
                    steer, vel = self._vel_cmd
                    enc = self._motor_data[3]
                    err = vel - enc
                    self._dt += time.time() - self._pd_time
                    d_err = (err - self._last_err) / self._dt
                    self._last_err = err
                    motor = self._p * err + self._d * d_err
                    if vel > 0:
                        if enc < 0.1:
                            motor = min(max(motor, 5), 15)
                        else:
                            motor = min(max(motor, 5), 20)
                    elif vel < 0:
                        if enc > -0.1:
                            motor = max(min(motor, -5), -15)
                        else:
                            motor = max(min(motor, -5), -20)
                    self._motor_cmd = (steer, motor)
                    self._dt = 0

                self._pd_time = time.time()

                self._last_motor_data = self._motor_data

            if self._is_calibrated and self._motor_cmd is not None:
                self._send_motor_cmd(self._motor_cmd)
                #        except:
                #            if self._motor_ser is not None and self._motor_ser.is_open:
                #                self._send_motor_cmd()
                #                self._motor_ser.close()

    def _imu_thread(self):
        try:
            while self._imu_ser.is_open:
                data = self._read_ser(self._imu_ser)
                if len(data) > 0 and data[0] == '(' and data[-3] == ')':
                    imu_data = data[5:-3].split(",")
                    if len(imu_data) == 6:
                        if self._default_defined:
                            self._imu_data = np.array(imu_data, dtype=np.float32) - self._default_imu_data
                        else:
                            self._default_imu_data = np.array(imu_data, dtype=np.float32)

                if self._is_calibrated:
                    if self._imu_data[2] + self._default_imu_data[2] < -9.0:
                        self._crashed = True
                        self._flip = True
                        print("flip crash")
                    elif self._motor_data[2] >= 1 and \
                                    self._imu_data[0] < -10.0:
                        self._crashed = True
                        self._flip = False
                        print("jolt crash")
                    elif self._motor_data[3] <= 0.7 and self._max_vel > 0.7 and self._motor_cmd is not None and \
                                    self._motor_cmd[1] >= 5.0:
                        self._crashed = True
                        self._flip = False
                        print("stuck crash")
                    else:
                        self._flip = False

        except:
            if self._imu_ser is not None and self._imu_ser.is_open:
                self._imu_ser.close()

    def _video_thread(self):
        try:
            while self._cam.isOpened():
                ret, data = self._cam.read()
                if ret:
                    self._image = np.array(data)[:, :, ::-1]
        except:
            if self._cam is not None and self._cam.isOpened():
                self._cam.release()

    # Get and set methods

    def get_motor_data(self):
        return self._motor_data

    def get_imu_data(self):
        return self._imu_data

    def get_image(self):
        return self._image

    def get_crash(self):
        return self._crashed

    def get_flip(self):
        return self._flip

    def set_motor_cmd(self, cmd):
        self._motor_cmd = cmd
        # Assumes that you are turning off vel_cmd
        self._vel_cmd = None

    def set_vel_cmd(self, cmd):
        self._vel_cmd = cmd
        self._new_vel_cmd = True
        self._pd_time = time.time()

    def set_pd(self, p, d):
        self._p = p
        self._d = d

    def reset_crash(self):
        self._crashed = False
        self._max_vel = 0

    # Utility functions

    def _read_ser(self, ser):
        return ser.readline().decode("utf-8")

    def _send_motor_cmd(self, motor_cmd=0):
        # Default value is the calibrated zero values
        if self._is_calibrated and motor_cmd is not None and motor_cmd != self._last_motor_cmd:
            #            print(motor_cmd)
            self._last_motor_cmd = motor_cmd
            steer, motor = 100 * np.clip(np.array(motor_cmd) + self._default_steer_motor, 0.0, 99.99)
            cmd_text = str.encode('(1{0:04.0f}{1:04.0f})'.format(steer, motor))
            self._motor_ser.write(cmd_text)

    def _setup_plot(self):
        plt.ion()
        self._fig, axs = plt.subplots(6, sharex=True)
        self._steer_axs = axs[0]
        axs[0].set_title('Steering')
        self._steer_plot, = axs[0].plot(self._times, self._steer_values)
        self._motor_axs = axs[1]
        axs[1].set_title('Motor')
        self._motor_plot, = axs[1].plot(self._times, self._motor_values)
        self._encoder_axs = axs[2]
        axs[2].set_title('Encoder')
        self._encoder_plot, = axs[2].plot(self._times, self._encoder_values)

        self._x_axs = axs[3]
        axs[3].set_title('X')
        self._x_plot, = axs[3].plot(self._times, self._x_values)
        self._y_axs = axs[4]
        axs[4].set_title('Y')
        self._y_plot, = axs[4].plot(self._times, self._y_values)
        self._z_axs = axs[5]
        axs[5].set_title('Z')
        self._z_plot, = axs[5].plot(self._times, self._z_values)

    def update_plot(self):

        if self._is_plotting:
            if self._is_calibrated:
                self._times.append(time.time() - self._start_time)
                self._steer_values.append(self._motor_data[1])
                self._motor_values.append(self._motor_data[2])
                self._encoder_values.append(self._motor_data[3])

                self._times = self._times[-100:]
                self._steer_values = self._steer_values[-100:]
                self._motor_values = self._motor_values[-100:]
                self._encoder_values = self._encoder_values[-100:]

                self._x_values.append(self._imu_data[0])
                self._y_values.append(self._imu_data[1])
                self._z_values.append(self._imu_data[2])

                self._x_values = self._x_values[-100:]
                self._y_values = self._y_values[-100:]
                self._z_values = self._z_values[-100:]

                if len(self._times) > 0:
                    max_time = self._times[-1]
                    min_time = self._times[0]
                    max_len = len(self._times)
                    self._steer_axs.set_xlim([min_time, max_time])

                    self._steer_plot.set_ydata(self._steer_values[:max_len])
                    self._steer_plot.set_xdata(self._times[:max_len])
                    self._steer_axs.set_ylim([np.min(self._steer_values) - 1, np.max(self._steer_values) + 1])

                    self._motor_plot.set_ydata(self._motor_values[:max_len])
                    self._motor_plot.set_xdata(self._times[:max_len])
                    self._motor_axs.set_ylim([np.min(self._motor_values) - 1, np.max(self._motor_values) + 1])

                    self._encoder_plot.set_ydata(self._encoder_values[:max_len])
                    self._encoder_plot.set_xdata(self._times[:max_len])
                    self._encoder_axs.set_ylim([np.min(self._encoder_values) - 1, np.max(self._encoder_values) + 1])

                    self._x_plot.set_ydata(self._x_values[:max_len])
                    self._x_plot.set_xdata(self._times[:max_len])
                    self._x_axs.set_ylim([np.min(self._x_values) - 1, np.max(self._x_values) + 1])

                    self._y_plot.set_ydata(self._y_values[:max_len])
                    self._y_plot.set_xdata(self._times[:max_len])
                    self._y_axs.set_ylim([np.min(self._y_values) - 1, np.max(self._y_values) + 1])

                    self._z_plot.set_ydata(self._z_values[:max_len])
                    self._z_plot.set_xdata(self._times[:max_len])
                    self._z_axs.set_ylim([np.min(self._z_values) - 1, np.max(self._z_values) + 1])

                    self._fig.canvas.draw()
                    plt.pause(0.0001)

    def calibrate(self):
        input('When you are done calibrating press Enter')
        while self._default_steer_motor is None or self._default_imu_data is None:
            pass
        self._default_defined = True
        while self._motor_data is None or self._imu_data is None:
            pass
        print('Calibrating done')
        self._start_time = time.time()
        self._is_calibrated = True

    def close(self):
        if self._motor_ser is not None:
            self._send_motor_cmd()
            self._motor_ser.close()
        if self._imu_ser is not None:
            self._imu_ser.close()


if __name__ == '__main__':
    handler = SensorsHandler(is_plotting=False)
    start = cur_time = time.time()
    #    handler.set_motor_cmd((0., 10.))
    #    handler.set_vel_cmd((0., 1.0))
    i = 0
    while time.time() - start < 120:
        if handler._is_plotting:
            if time.time() - cur_time > 0.01:
                cur_time = time.time()
                handler.update_plot()
        else:
            if handler.get_crash():
                print("crashed {0}".format(i))
                i += 1
                handler.reset_crash()
                #        image = handler.get_image()
                #        if image is not None:
                #            plt.imshow(image)
                #            plt.show()
        if time.time() - cur_time > 0.025:
            cur_time = time.time()
            #            print(handler.get_motor_data()[3])
            #            print(handler.get_imu_data()[3:])
            #        pass
    handler.set_motor_cmd((0.0, 0.0))
    handler.close()