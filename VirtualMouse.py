import mediapipe as mp
import cv2
import pyautogui
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1  # Track only one hand for simplicity
        )
        self.screen_width, self.screen_height = pyautogui.size()

    def track_hand(self):
        cap = cv2.VideoCapture(0)
        x_smooth, y_smooth = 0, 0
        smoothing_factor = 0.5
        scrolling = False
        zooming = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.flip(image_rgb, 1)

            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                
                hand_landmarks = results.multi_hand_landmarks[0]  # Use the first detected hand

                # Extract index and middle finger tip landmarks
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                # Convert landmarks to screen coordinates
                index_x, index_y = int(index_tip.x * self.screen_width), int(index_tip.y * self.screen_height)
                middle_x, middle_y = int(middle_tip.x * self.screen_width), int(middle_tip.y * self.screen_height)
                thumb_x, thumb_y = int(thumb_tip.x * self.screen_width), int(thumb_tip.y * self.screen_height)

                # Calculate finger distances
                finger_distance = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
                thumb_distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)

                # Smoothen mouse movement
                x_smooth = int(smoothing_factor * index_x + (1 - smoothing_factor) * x_smooth)
                y_smooth = int(smoothing_factor * index_y + (1 - smoothing_factor) * y_smooth)

                # Move the mouse pointer within the screen bounds
                x_smooth = max(0, min(self.screen_width, x_smooth))
                y_smooth = max(0, min(self.screen_height, y_smooth))

                # Check for scrolling and zooming gestures
                if finger_distance < 80:
                    scrolling = True
                else:
                    scrolling = False

                if thumb_distance < 50:
                    zooming = True
                else:
                    zooming = False

                # Perform actions based on gestures
                if scrolling:
                    pyautogui.scroll(1)  # Scroll down
                elif zooming:
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(-1)  # Zoom out
                    pyautogui.keyUp('ctrl')

                pyautogui.moveTo(x_smooth, y_smooth)

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Virtual Mouse', image_bgr)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.track_hand()
