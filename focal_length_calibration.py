#!/usr/bin/env python
# coding: utf-8

import dlib
import cv2
import numpy as np
import pygame

d = 50
P_IPD = 6.3
video_res = [640, 480]
cursor_radius = 10  # Define the cursor radius

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat")

face_detect_size = [320, 240]

def get_eye_pos(shape, pos="L"):
    if pos == "R":
        lc = 36
        rc = 39
        FP_seq = [36, 37, 38, 39, 40, 41]
    elif pos == "L":
        lc = 42
        rc = 45
        FP_seq = [45, 44, 43, 42, 47, 46]
    else:
        print("Error: Wrong pos parameter")

    eye_cx = (shape.part(rc).x + shape.part(lc).x) * 0.5
    eye_cy = (shape.part(rc).y + shape.part(lc).y) * 0.5
    eye_center = [eye_cx, eye_cy]
    eye_len = np.absolute(shape.part(rc).x - shape.part(lc).x)
    bx_d5w = eye_len * 3 / 4
    bx_h = 1.5 * bx_d5w

    sft_up = bx_h * 7 / 12
    sft_low = bx_h * 5 / 12
    E_TL = (int(eye_cx - bx_d5w), int(eye_cy - sft_up))
    E_RB = (int(eye_cx + bx_d5w), int(eye_cy + sft_low))
    return eye_center, E_TL, E_RB

pygame.init()

display_width = video_res[0]
display_height = video_res[1]
display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Eye Gaze Cursor")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

cursor_color = (255, 0, 0)  # Red color for the cursor

pygame.mixer.init()
music = pygame.mixer.Sound("music.wav")

cap = cv2.VideoCapture(0)

running = True
while running:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray, (face_detect_size[0], face_detect_size[1]))
        detections = detector(face_detect_gray, 0)
        x_ratio = video_res[0] / face_detect_size[0]
        y_ratio = video_res[1] / face_detect_size[1]

        display.fill(BLACK)  # Clear the display

        for k, bx in enumerate(detections):
            target_bx = dlib.rectangle(left=int(bx.left() * x_ratio), right=int(bx.right() * x_ratio),
                                       top=int(bx.top() * y_ratio), bottom=int(bx.bottom() * y_ratio))

            shape = predictor(gray, target_bx)
            LE_center, _, _ = get_eye_pos(shape, pos="L")
            RE_center, _, _ = get_eye_pos(shape, pos="R")

            avg_eye_x = int((LE_center[0] + RE_center[0]) / 2)
            avg_eye_y = int((LE_center[1] + RE_center[1]) / 2)

            mouth_points = [] 
            for i in range(48, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                mouth_points.append((x, y))

            mouth_width = np.sqrt((mouth_points[3][0] - mouth_points[9][0]) ** 2 + (mouth_points[3][1] - mouth_points[9][1]) ** 2)
            mouth_open_threshold = 30
            max_mouth_width = 100  # Set the maximum mouth width for color and size changes

            if mouth_width > mouth_open_threshold:
                mouth_open = True
                if not music.get_num_channels():
                    music.play()

                # Calculate cursor color and size based on mouth width
                cursor_size = int(cursor_radius * (mouth_width / max_mouth_width))
                cursor_color_value = int(255 * (mouth_width / max_mouth_width))
                cursor_color = (cursor_color_value, 255 - cursor_color_value, 0)  # Transition from green to red
            else:
                mouth_open = False
                music.stop()
                cursor_size = cursor_radius
                cursor_color = (0, 255, 0)  # Green color for closed mouth

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            display.blit(pygame.transform.rotate(frame_surface, 270), (0, 0))  # Rotate the frame surface by 180 degrees
            pygame.draw.circle(display, cursor_color, (avg_eye_x, avg_eye_y), cursor_size)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

    else:
        running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
quit()