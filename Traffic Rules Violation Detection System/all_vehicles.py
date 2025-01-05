import cv2
import dlib
import time
import math
import helm

carCascade = cv2.CascadeClassifier('cars.xml')
bikeCascade = cv2.CascadeClassifier('motor-v4.xml')
video = cv2.VideoCapture('record.mkv')

LAG = 7
WIDTH = 1280
HEIGHT = 720
OPTIMISE = 7

def estimateSpeed(location1, location2, fps):
    d_pixels = math.sqrt(
        math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2)
    )
    ppm = 8.8
    d_meters = d_pixels / ppm
    if fps == 0.0:
        fps = 18
    speed = d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    identity = [0 for _ in range(1000)]
    Helmets = ["No Helmet Detected" for _ in range(1000)]

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter += 1

        carIDtoDelete = []
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar, y_bar = x + 0.5 * w, y + 0.5 * h

                matchCarID = None
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x, t_y, t_w, t_h = (
                        int(trackedPosition.left()),
                        int(trackedPosition.top()),
                        int(trackedPosition.width()),
                        int(trackedPosition.height()),
                    )
                    t_x_bar, t_y_bar = t_x + 0.5 * t_w, t_y + 0.5 * t_h

                    if (
                        (t_x <= x_bar <= (t_x + t_w))
                        and (t_y <= y_bar <= (t_y + t_h))
                        and (x <= t_x_bar <= (x + w))
                        and (y <= t_y_bar <= (y + h))
                    ):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y, t_w, t_h = (
                int(trackedPosition.left()),
                int(trackedPosition.top()),
                int(trackedPosition.width()),
                int(trackedPosition.height()),
            )
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    result = False
                    roi = resultImage[y1 : y1 + h1, x1 : x1 + w1]
                    if speed[i] is None:
                        speed[i] = estimateSpeed(
                            [x1, y1, w1, h1], [x2, y2, w2, h2], 18
                        )
                        cv2.putText(
                            resultImage,
                            f"Speed: {int(speed[i])} km/hr",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2,
                        )

        cv2.imshow("result", resultImage)
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    trackMultipleObjects()
