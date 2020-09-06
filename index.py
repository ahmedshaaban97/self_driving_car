# from fastai.vision import *
# from PIL.Image import Image
# folder = 'bottles'
# file = 'bottles.zip'
#
# folder = 'cups'
# file = 'cups.zip'
#
# path = Path('data/objects')
# dest = path/folder
# classes = ['cups','bottles',]
#
# #print(path/folder/file)
# for c in classes:
#     print(c)
#     verify_images(path/c, delete=True, max_size=500)
#
# np.random.seed(42)
# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#
#
# #print(data.classes)
# learn = cnn_learner(data, models.resnet34, metrics=error_rate)
#
# #learn.unfreeze()
#
# #learn.lr_find()
#
# #learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-3))
#
# #learn.save('stage-2')
#
#
# learn.load('stage-2');
# img = open_image(path/'bottles'/'1.jpg')
# #img = Image(img)
# prediction = learn.predict(img)
# print(prediction[0])
#
# if str(prediction[0]) == 'bottles':
#     print('we did it man')
#
# # img = learn.data.train_ds[0][0]
# # #print(learn.predict(img))
# # print(path/'cups'/'cup1.jpg')
#
#
#
# #print(img.data.shape)
#


#
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('test3b.jpg',0)
#
# height = img.shape[0]
# width = img.shape[1]
#
# edges = cv.Canny(img,160,200)
#
# for i in range(int(height/2)+10):
#     for j in range(width):
#         edges[i][j]=255
#
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
#
# plt.show()


###

# import numpy as np
# import cv2
#
# inputImage = cv2.imread("test1b.jpg")
#
# height = inputImage.shape[0]
# width = inputImage.shape[1]
#
#
#
# inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
#
#
#
#
#
#
#
#
# edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
#
#
# for i in range(int(height/2)):
#      for j in range(width):
#          edges[i][j]=255
#
#
#
# minLineLength = 30
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
#         pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
#         cv2.polylines(inputImage, [pts], True, (0,255,0))
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)
# cv2.imshow("Trolley_Problem_Result", inputImage)
# cv2.imshow('edge', edges)
# cv2.waitKey(0)
# #
# #



import numpy as np
import cv2
import logging
import serial
import time
import requests as req
import json
import math

def region_of_interest1(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (width*1/3,height*1/3),
        (width*1/3,height*2/3),
        (width*2/3,height*2/3),
        (width*2/3,height*1/3),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("cropped", cropped_edges)
    return cropped_edges

def detect_edges(frame):
    # height = inputImage.shape[0]
    # width = inputImage.shape[1]
    inputImageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", inputImageHSV)
    lower_green = np.array([80, 40, 40])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(inputImageHSV, lower_green, upper_green)
    # cv2.imshow("blue mask", mask)
    edges = cv2.Canny(mask, 200, 400)
    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("cropped", cropped_edges)
    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(s, frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        #s.write(b"l")
        time.sleep(0.25)
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def detect_lane(s, frame):
    edges = detect_edges(frame)
    ROI = region_of_interest(edges)
    line_segments = detect_line_segments(ROI)
    lane_lines = average_slope_intercept(s, frame, line_segments)

    return lane_lines


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    time.sleep(0.5)
    return line_image


def compute_steering_angle(s, frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
                    We assume that camera is calibrated to point to dead center
                """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        print('No Lanes')
        s.write(b"3")
        time.sleep(0.25)
        # Do Nothing
    else:
        height, width, _ = frame.shape
        if len(lane_lines) == 1:
            logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
        else:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            camera_mid_offset_percent = 0.02  # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid
        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)
        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
        return steering_angle

def send_motion_commands(s, steering_angle):
    if steering_angle >= 50 and steering_angle <= 140:
        s.write(b"1")
        print("forward")
        time.sleep(0.3)
    elif steering_angle < 50:
        s.write(b"3")
        print("left")
        time.sleep(0.1)
    else:
        s.write(b"4")
        print("right")
        time.sleep(0.2)




        # s.write(b"b")
        # print("backward!")
        # time.sleep(5)
        #
        # s.write(b"s")
        # print("stop")
        # time.sleep(5)

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def HSV_edges_masks(inputImage):
    inputImageHSV = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    lower_green = np.array([80, 40, 40])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(inputImageHSV, lower_green, upper_green)
    edges = cv2.Canny(mask, 200, 400)
    return mask, edges


def get_Contour(inputImage):
    ###Contour
    src_gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    ## [Canny]
    # Detect edges using Canny
    src_gray = region_of_interest1(src_gray)
    threshold = 200
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    cv2.imshow("cannyout", canny_output)
    ## [Canny]
    ## [findContours]
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []
    ## [findContours]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(
            contours_poly[i]
        )  # x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        areaArray.append(area)
    ## [zeroMat]
    # print("coor", boundRect[i])
    drawing = np.zeros(
        (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8
    )
    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    direction = None
    if (len(contours) >= 3):

        # find the nth largest contour [n-1][1], in this case 2
        secondlargestcontour = sorteddata[2][1]

        # draw it
        x, y, w, h = cv2.boundingRect(secondlargestcontour)
        cv2.drawContours(drawing, secondlargestcontour, -1, (255, 0, 0), 2)
        cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(x, y, w, h)
        if (x < 170):
            direction = 1  #right
        else:
            direction = 2  #left
    return drawing, direction


def main():
    url = "http://192.168.43.160:8080/shot.jpg"
    resp = req.get("http://b2134fde.ngrok.io/getRoom")
    print("I am not in yet !!")
    json_string = json.loads(resp.text)

    if json_string['t'] == 1:
        s = serial.Serial('COM13', 9600, timeout=1)
        print('hello from the other side')
        while 1:
              # choose the outgoing one
                # print("connected!")
                #time.sleep(2)
                img_resp = req.get(url)
                # # print(type(img_resp))
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                inputImage = cv2.imdecode(img_arr, -1)

                # mask, edges = HSV_edges_masks(inputImage)
                # drawing, direction = get_Contour(inputImage)
                #
                #
                # drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
                # # print(edges.shape,"hiiiiiii", drawing_gray.shape)
                # combined = cv2.add(drawing_gray, edges)
                #
                # combined_pic = cv2.add(drawing, inputImage)  # overlay contour on src


                lane_lines = detect_lane(s, inputImage)
                lane_lines_image = display_lines(inputImage, lane_lines)
                steering = compute_steering_angle(s, inputImage, lane_lines)
                if steering == None:
                    s.write(b"3")
                    print("none")
                    time.sleep(0.25)
                else:
                    heading_line_image = display_heading_line(inputImage, steering, (0, 0, 255), 5, )
                    send_motion_commands(s, steering)
                    cv2.imshow("Heading Line", heading_line_image)
                # else:
                #     if direction == 1:
                #         s.write(b"r")
                #         print("Right Object")
                #         time.sleep(0.25)
                #         #s.write(b"f")
                #         direction = None
                #     else:
                #         s.write(b"l")
                #         print("Left Object")
                #         time.sleep(0.25)
                #         #s.write(b"f")
                #         direction = None
                cv2.imshow("AndroidCam", inputImage)
                cv2.imshow("Lane Lines", lane_lines_image)
                # cv2.imshow("combined", combined_pic)
                # cv2.imshow("Contours", drawing)

                if cv2.waitKey(1) == 27:
                    break


main()
