import cv2
import mediapipe as mp
import math

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Take live camera input for pose detection
cap = cv2.VideoCapture(0)

# Set the desired frame rate (e.g., 2 frames per second)
frame_rate = 2  # Adjust as needed

# Threshold for choking detection (adjust as needed)
choking_threshold = 0.2  # You may need to fine-tune this value

# Read each frame/image from the capture object
while True:
    ret, img = cap.read()
    # Resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (600, 400))

    # Do Pose detection
    results = pose.process(img)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Access specific landmarks (e.g., neck and wrists)
        neck = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate the distance between neck and wrists
        left_wrist_distance = math.sqrt((left_wrist.x - neck.x) ** 2 + (left_wrist.y - neck.y) ** 2)
        right_wrist_distance = math.sqrt((right_wrist.x - neck.x) ** 2 + (right_wrist.y - neck.y) ** 2)

        # Check for choking action based on threshold
        if left_wrist_distance < choking_threshold and right_wrist_distance < choking_threshold:
            print("Choking action detected")

    # Draw the detected pose on the original video/live stream
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )

    # Display pose on the original video/live stream
    cv2.imshow("Pose Estimation", img)

    cv2.waitKey(1)

    # Add a delay to control the frame rate
    # time.sleep(1.0 / frame_rate)  # Sleep for the desired frame rate (e.g., 1/2 seconds)
