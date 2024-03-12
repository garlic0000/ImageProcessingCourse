import cv2 as cv
import numpy as np


def readcamera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Can not open camera")
        exit()
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_index = 0
    cv.namedWindow('Camera playing', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_index += 1
        if not ret:
            print("Can't receive frame (stream end?), Exiting...")
            break
        img = np.flip(frame, 1)
        img_montage = np.hstack((frame, img))
        cv.imshow('Camera playing', img_montage)
        keycode = cv.waitKey(0)
        if keycode == 27:
            break
        elif keycode == ord('s'):
            frame_name = "../Picture/camera_frame_{}.jpg".format(frame_index)
            cv.imwrite(frame_name, frame)
        cap.release()
        cv.destroyAllWindows()


def readvideo():
    cap = cv.VideoCapture("../Video/streetlamp.mp4")
    if not cap.isOpened():
        print("Cannot open the file")
        exit()
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_gray = cv.VideoWriter('video_gray.avi', fourcc, int(fps), (int(frame_width), int(frame_height)), isColor=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame, or end of file playback, Exiting...")
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        out_gray.write(gray_frame)
        cv.imshow('Video color frame', frame)
        cv.imshow('Video gray frame', gray_frame)
        if cv.waitKey(int(1000 / fps)) & 0xFF == 27:
            break
    cap.release()
    out_gray.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    readcamera()
    # readvideo()
