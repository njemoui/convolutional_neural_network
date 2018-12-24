import cv2
def Frame_Capture(path):
    vid_obj = cv2.VideoCapture(path)
    count = 0
    success = 1

    while success:
        success, image = vid_obj.read()
        cv2.imwrite("data/frames/wake/frame%d.jpg" % count, image)
        count += 1
if __name__ == '__main__':
    Frame_Capture("data/videos/wake_2.mp4")