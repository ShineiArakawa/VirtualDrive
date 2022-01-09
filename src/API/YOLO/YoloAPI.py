from API.YOLO.yolo import YOLO

import os
import cv2


class YoloAPI:
    NETWORK_TYPE = "prn"
    # NETWORK_TYPE = "v4-tiny"
    # SIZE = 512
    SIZE = 256
    CONFIDENCE = 0.1
    MAX_NUM_HAND = 2

    def __init__(self, cameraID: int) -> None:
        yoloWorkingDirPath = os.path.dirname(__file__)

        self.__yolo = None
        self.__videoCapture = None
        self.__rval = None
        self.__frame = None
        self.__videoHeight = None
        self.__videoWidth = None

        if self.NETWORK_TYPE == "normal":
            print("loading yolo...")
            pathConf = os.path.join(yoloWorkingDirPath, "models/cross-hands.cfg")
            pathWeight = os.path.join(yoloWorkingDirPath, "models/cross-hands.weights")
            self.__yolo = YOLO(pathConf, pathWeight, ["hand"])
        elif self.NETWORK_TYPE == "prn":
            print("loading yolo-tiny-prn...")
            pathConf = os.path.join(yoloWorkingDirPath, "models/cross-hands-tiny-prn.cfg")
            pathWeight = os.path.join(yoloWorkingDirPath, "models/cross-hands-tiny-prn.weights")
            self.__yolo = YOLO(pathConf, pathWeight, ["hand"])
        elif self.NETWORK_TYPE == "v4-tiny":
            print("loading yolov4-tiny-prn...")
            pathConf = os.path.join(yoloWorkingDirPath, "models/cross-hands-yolov4-tiny.cfg")
            pathWeight = os.path.join(yoloWorkingDirPath, "models/cross-hands-yolov4-tiny.weights")
            self.__yolo = YOLO(pathConf, pathWeight, ["hand"])
        else:
            print("loading yolo-tiny...")
            pathConf = os.path.join(yoloWorkingDirPath, "models/cross-hands-tiny.cfg")
            pathWeight = os.path.join(yoloWorkingDirPath, "models/cross-hands-tiny.weights")
            self.__yolo = YOLO(pathConf, pathWeight, ["hand"])

        self.__yolo.size = int(self.SIZE)
        self.__yolo.confidence = float(self.CONFIDENCE)

        print("starting webcam...")
        cv2.namedWindow("preview")
        self.__videoCapture = cv2.VideoCapture(cameraID)

        if self.__videoCapture.isOpened():  # try to get the first frame
            self.__rval, self.__frame = self.__videoCapture.read()
        else:
            self.__rval = False

        self.__videoWidth = self.__videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.__videoHeight = self.__videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Capture: Height= {self.__videoHeight}, width= {self.__videoWidth}")

    def getHandPsition(self):
        positions = list()

        width, height, inference_time, results = self.__yolo.inference(
            self.__frame)

        # display fps
        try:
            cv2.putText(self.__frame, f'{round(1/inference_time,2)} FPS',
                        (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            # print(e)
            pass

        # sort by confidence
        results.sort(key=lambda x: x[2])

        # display hands
        for detection in results[:self.MAX_NUM_HAND]:
            id, name, confidence, x, y, w, h = detection
            cx = (x + (w / 2)) / self.__videoWidth
            cy = (y + (h / 2))  / self.__videoHeight
            positions.append((cx, cy))

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(self.__frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(self.__frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        self.__frame
        cv2.imshow("preview", self.__frame)

        self.__rval, self.__frame = self.__videoCapture.read()

        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            raise KeyboardInterrupt
        
        return positions

    def __del__(self):
        cv2.destroyWindow("preview")
        self.__videoCapture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
