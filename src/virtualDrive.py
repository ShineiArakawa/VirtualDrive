from API.Carla.CarlaAPI import CarlaAPI
from API.YOLO.YoloAPI import YoloAPI
from Controller.Controller import MainController


def main():
    CAMERA_ID = 1

    carlaAPI = CarlaAPI()
    # carlaAPI = None
    yoloAPI = YoloAPI(cameraID=CAMERA_ID)
    controller = MainController(carlaAPI=carlaAPI, yoloAPI=yoloAPI)

    controller.start()

if __name__ == "__main__":
    main()