from cv2 import stereoCalibrate
from API.Carla.CarlaAPI import CarlaAPI
from API.YOLO.YoloAPI import YoloAPI
from API.Carla.core import World

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging


class MainController:
    # autopep8: off
    # LOG_LEVEL                      = logging.DEBUG
    LOG_LEVEL                      = logging.INFO

    THRESHOULD_DIFF_STEERING_ANGLE = 45.0
    # autopep8: on

    def __init__(self, carlaAPI: CarlaAPI, yoloAPI: YoloAPI) -> None:
        self.__logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=self.LOG_LEVEL)

        self.__carla: CarlaAPI = carlaAPI
        self.__yolo: YoloAPI = yoloAPI
        self.__steeringAngle: float = 0.0
        self.__steeringLog: List[float] = None
        pass

    def start(self):
        self.__steeringLog = []
        try:
            while True:
                # hand detection
                position = self.__yolo.getHandPsition()
                self.__getSteeringAngle(position)
                self.__steeringLog.append(self.__steeringAngle)
                # self.__logger.info(f"steering angle= {self.__steeringAngle}")

                # drive
                self.__carla.drive(steeringAngle=self.__steeringAngle)

        except KeyboardInterrupt:
            self.__plotSteeringLog()
            self.__exitProgram()

    def __getSteeringAngle(self, potisions: List[Tuple[float]]):
        nDetectedHands = len(potisions)
        self.__steeringAngle = 0.0

        if nDetectedHands > 1:
            rightHandPosition = potisions[0]
            leftHandPosition = potisions[1]

            if leftHandPosition[0] < rightHandPosition[1]:
                rightHandPosition = potisions[1]
                leftHandPosition = potisions[0]

            try:
                inclination = (leftHandPosition[1] - rightHandPosition[1]) / \
                    (leftHandPosition[0] - rightHandPosition[0])
                tmpSteeringAngle = np.rad2deg(np.arctan(inclination))

                if np.abs(tmpSteeringAngle - self.__steeringAngle) < self.THRESHOULD_DIFF_STEERING_ANGLE:
                    self.__steeringAngle = tmpSteeringAngle
            except ZeroDivisionError:
                pass

    def __plotSteeringLog(self):
        rolling = World.LENGTH_OF_STEERING_CACHE
        self.__steeringLog = np.convolve(
            self.__steeringLog, np.ones(rolling), 'valid') / rolling
        x = range(len(self.__steeringLog))

        plt.title(f"rolling= {rolling}")
        plt.xlabel("time")
        plt.ylabel("steering angle")
        plt.grid()
        plt.plot(x, self.__steeringLog)
        plt.show()

    def __exitProgram(self):
        del self.__yolo
        del self.__carla

        print("Bye!")
        quit(0)
