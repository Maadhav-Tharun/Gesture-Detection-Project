import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FILE SETTINGS
# ============================================================

MODEL_FILE = "gesture_model.pkl"
HAND_MODEL_FILE = "hand_landmarker.task"


# ============================================================
# ACCURACY SETTINGS
# ============================================================

# Minimum confidence required
MIN_CONFIDENCE = 80.0

# Difference between best and second-best prediction
MIN_CONFIDENCE_GAP = 15.0

# Number of recent predictions used for smoothing
SMOOTHING_FRAMES = 10

# Minimum number of matching predictions required
# before accepting a gesture
MIN_VOTES = 6

# Number of consecutive valid predictions required
# before changing displayed gesture
STABLE_FRAMES_REQUIRED = 3


# ============================================================
# CAMERA SETTINGS
# ============================================================

# DroidCam is usually 1, 2, 3 etc.
# 0 is normally the laptop camera.

CAMERA_INDEXES_TO_TEST = [1, 2, 3, 4, 5, 0]


# ============================================================
# LOAD GESTURE MODEL
# ============================================================

print("=" * 60)
print("HIGH ACCURACY GESTURE RECOGNITION SYSTEM")
print("=" * 60)

print("\nLoading gesture model...")

try:

    model = joblib.load(MODEL_FILE)

    print("SUCCESS: gesture_model.pkl loaded")

    print("\nTrained gestures:")

    for gesture in model.classes_:
        print("   ", gesture)

except Exception as error:

    print("\nERROR: Could not load gesture_model.pkl")
    print(error)

    exit()


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

try:

    expected_features = model.n_features_in_

    print(
        "\nModel expects:",
        expected_features,
        "features"
    )

    if expected_features != 126:

        print("\nWARNING!")
        print("This model does not contain 126 features.")
        print("Make sure you trained it using the latest trainer.")

except Exception:

    pass


# ============================================================
# LOAD MEDIAPIPE HAND LANDMARKER
# ============================================================

print("\nLoading MediaPipe Hand Landmarker...")

try:

    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL_FILE
    )

    options = vision.HandLandmarkerOptions(

        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_hands=2,

        # Slightly stronger detection
        min_hand_detection_confidence=0.60,

        min_hand_presence_confidence=0.60,

        min_tracking_confidence=0.60
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    print("SUCCESS: Hand Landmarker loaded")

except Exception as error:

    print("\nERROR: Could not load hand_landmarker.task")
    print(error)

    exit()


# ============================================================
# EXTRACT ONE HAND
# ============================================================

def extract_hand(hand_landmarks):

    """
    Extract 63 normalized features from one hand.

    21 landmarks × 3 coordinates = 63 features.

    Steps:
    1. Make wrist the origin.
    2. Normalize hand size.
    """

    wrist = hand_landmarks[0]

    relative_points = []

    # --------------------------------------------------------
    # STEP 1: Wrist-relative coordinates
    # --------------------------------------------------------

    for landmark in hand_landmarks:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        relative_points.append([x, y, z])


    # --------------------------------------------------------
    # STEP 2: Calculate hand size
    # --------------------------------------------------------

    distances = []

    for point in relative_points:

        distance = np.sqrt(
            point[0] ** 2 +
            point[1] ** 2 +
            point[2] ** 2
        )

        distances.append(distance)


    # --------------------------------------------------------
    # STEP 3: Normalize hand size
    # --------------------------------------------------------

    hand_size = max(distances)

    if hand_size < 0.0001:

        hand_size = 1.0


    # --------------------------------------------------------
    # STEP 4: Create normalized features
    # --------------------------------------------------------

    features = []

    for point in relative_points:

        features.append(point[0] / hand_size)
        features.append(point[1] / hand_size)
        features.append(point[2] / hand_size)


    return features


# ============================================================
# EXTRACT TWO HANDS
# ============================================================

def extract_two_hands(result):

    """
    63 features = Left hand
    63 features = Right hand

    Total = 126 features.
    """

    left_hand = [0.0] * 63

    right_hand = [0.0] * 63


    if not result.hand_landmarks:

        return left_hand + right_hand


    for i, landmarks in enumerate(
        result.hand_landmarks
    ):

        hand_name = (
            result.handedness[i][0].category_name
        )


        hand_data = extract_hand(
            landmarks
        )


        if hand_name == "Left":

            left_hand = hand_data


        elif hand_name == "Right":

            right_hand = hand_data


    return left_hand + right_hand


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]


    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)
    ]


    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in connections:

        p1 = landmarks[start]
        p2 = landmarks[end]


        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # Draw points
    # --------------------------------------------------------

    for point in landmarks:

        x = int(point.x * width)
        y = int(point.y * height)


        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# ============================================================
# DROIDCAM SETUP
# ============================================================

print("\n" + "=" * 60)
print("DROIDCAM CAMERA SETUP")
print("=" * 60)

print("""
Make sure:

1. Phone is connected through USB.
2. USB Debugging is enabled.
3. DroidCam Client is open.
4. USB mode is selected.
5. Your phone is selected.
6. Video is checked.
7. Click START in DroidCam.
""")


input(
    "After DroidCam starts showing the phone camera, "
    "press ENTER here..."
)


# ============================================================
# SEARCH FOR DROIDCAM
# ============================================================

print("\nSearching for DroidCam camera...")

cap = None
selected_camera = None


for index in CAMERA_INDEXES_TO_TEST:

    print(
        f"Testing camera index {index}..."
    )


    test_cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW
    )


    if not test_cap.isOpened():

        test_cap.release()

        continue


    test_cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    test_cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    success, test_frame = test_cap.read()


    if success and test_frame is not None:

        print(
            f"Camera {index} is available."
        )


        # Prefer DroidCam
        if index != 0:

            cap = test_cap

            selected_camera = index

            break


    test_cap.release()


# ============================================================
# CAMERA NOT FOUND
# ============================================================

if cap is None:

    print("\n" + "=" * 60)
    print("DROIDCAM CAMERA NOT FOUND")
    print("=" * 60)

    print("""
Python could not find the DroidCam virtual camera.

Make sure DroidCam Client successfully starts first.

Try:

1. Disconnect USB.
2. Reconnect USB.
3. Enable USB Debugging.
4. Select USB mode in DroidCam.
5. Select your phone.
6. Click Refresh.
7. Click Start.
8. Then run this program again.
""")

    hand_landmarker.close()

    exit()


# ============================================================
# CAMERA SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("DROIDCAM FOUND")
print("=" * 60)

print(
    "Camera index:",
    selected_camera
)

print("Phone camera connected successfully.")

print("\nPress Q to quit.")


# ============================================================
# CAMERA RESOLUTION
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "High Accuracy Gesture Recognition"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# ============================================================
# TIMESTAMP
# ============================================================

last_timestamp = 0


# ============================================================
# SMOOTHING VARIABLES
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)


current_displayed_gesture = "No Hand"

candidate_gesture = None

candidate_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # ========================================================
    # READ CAMERA
    # ========================================================

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Could not read DroidCam frame."
        )

        break


    # ========================================================
    # MIRROR IMAGE
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = int(
        time.monotonic() * 1000
    )


    if timestamp <= last_timestamp:

        timestamp = last_timestamp + 1


    last_timestamp = timestamp


    # ========================================================
    # HAND DETECTION
    # ========================================================

    try:

        result = hand_landmarker.detect_for_video(
            mp_image,
            timestamp
        )

    except Exception as error:

        print(
            "MediaPipe error:",
            error
        )

        continue


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    confidence = 0.0

    confidence_gap = 0.0

    hands_detected = 0


    # ========================================================
    # NO HAND
    # ========================================================

    if not result.hand_landmarks:

        hands_detected = 0

        prediction_history.clear()

        current_displayed_gesture = "No Hand"

        candidate_gesture = None

        candidate_count = 0


    # ========================================================
    # HAND DETECTED
    # ========================================================

    else:

        hands_detected = len(
            result.hand_landmarks
        )


        # ----------------------------------------------------
        # DRAW HANDS
        # ----------------------------------------------------

        for landmarks in result.hand_landmarks:

            draw_hand(
                frame,
                landmarks
            )


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        features = extract_two_hands(
            result
        )


        # ----------------------------------------------------
        # NUMPY ARRAY
        # ----------------------------------------------------

        features_array = np.array(
            features,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        try:

            probabilities = model.predict_proba(
                features_array
            )[0]

        except Exception as error:

            print(
                "Prediction error:",
                error
            )

            continue


        # ----------------------------------------------------
        # SORT RESULTS
        # ----------------------------------------------------

        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        # ----------------------------------------------------
        # BEST PREDICTION
        # ----------------------------------------------------

        best_index = sorted_indices[0]


        best_confidence = (
            probabilities[best_index] * 100
        )


        predicted_label = model.classes_[
            best_index
        ]


        # ----------------------------------------------------
        # SECOND BEST
        # ----------------------------------------------------

        if len(sorted_indices) > 1:

            second_index = sorted_indices[1]


            second_confidence = (
                probabilities[second_index] * 100
            )


            confidence_gap = (
                best_confidence
                -
                second_confidence
            )

        else:

            confidence_gap = best_confidence


        confidence = best_confidence


        # ====================================================
        # FIRST ACCURACY CHECK
        # ====================================================

        if (

            best_confidence >= MIN_CONFIDENCE

            and

            confidence_gap >= MIN_CONFIDENCE_GAP

        ):

            valid_prediction = str(
                predicted_label
            )

        else:

            valid_prediction = "WRONG GESTURE"


        # ====================================================
        # ADD TO HISTORY
        # ====================================================

        prediction_history.append(
            valid_prediction
        )


        # ====================================================
        # MAJORITY VOTING
        # ====================================================

        if len(prediction_history) >= 5:

            vote_counter = Counter(
                prediction_history
            )


            most_common = (
                vote_counter
                .most_common(1)
            )


            majority_gesture = (
                most_common[0][0]
            )

            majority_votes = (
                most_common[0][1]
            )


            # ------------------------------------------------
            # Accept only if enough frames agree
            # ------------------------------------------------

            if majority_votes >= MIN_VOTES:

                # ============================================
                # WRONG GESTURE
                # ============================================

                if majority_gesture == "WRONG GESTURE":

                    current_displayed_gesture = (
                        "WRONG GESTURE"
                    )

                    candidate_gesture = None

                    candidate_count = 0


                # ============================================
                # VALID GESTURE
                # ============================================

                else:

                    # Same as current
                    if (
                        majority_gesture
                        ==
                        current_displayed_gesture
                    ):

                        candidate_gesture = None

                        candidate_count = 0


                    else:

                        # New candidate
                        if (
                            candidate_gesture
                            ==
                            majority_gesture
                        ):

                            candidate_count += 1

                        else:

                            candidate_gesture = (
                                majority_gesture
                            )

                            candidate_count = 1


                        # ====================================
                        # STABILITY CHECK
                        # ====================================

                        if (
                            candidate_count
                            >=
                            STABLE_FRAMES_REQUIRED
                        ):

                            current_displayed_gesture = (
                                majority_gesture
                            )

                            candidate_gesture = None

                            candidate_count = 0


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    PANEL_X = 15
    PANEL_Y = 15

    PANEL_WIDTH = 335
    PANEL_HEIGHT = 155


    # Black panel

    cv2.rectangle(

        frame,

        (
            PANEL_X,
            PANEL_Y
        ),

        (
            PANEL_X + PANEL_WIDTH,
            PANEL_Y + PANEL_HEIGHT
        ),

        (0, 0, 0),

        -1
    )


    # ========================================================
    # TEXT COLOR
    # ========================================================

    if current_displayed_gesture == "WRONG GESTURE":

        text_color = (
            0,
            0,
            255
        )


    elif current_displayed_gesture == "No Hand":

        text_color = (
            255,
            255,
            255
        )


    else:

        text_color = (
            0,
            255,
            0
        )


    # ========================================================
    # GESTURE TITLE
    # ========================================================

    cv2.putText(

        frame,

        "GESTURE:",

        (30, 42),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ========================================================
    # GESTURE RESULT
    # ========================================================

    cv2.putText(

        frame,

        str(current_displayed_gesture),

        (30, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.68,

        text_color,

        2
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    cv2.putText(

        frame,

        f"Confidence: {confidence:.1f}%",

        (30, 96),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # GAP
    # ========================================================

    cv2.putText(

        frame,

        f"Gap: {confidence_gap:.1f}%",

        (175, 96),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # HANDS
    # ========================================================

    cv2.putText(

        frame,

        f"Hands: {hands_detected}",

        (30, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # CAMERA
    # ========================================================

    cv2.putText(

        frame,

        f"DroidCam: {selected_camera}",

        (175, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # SMOOTHING INFO
    # ========================================================

    cv2.putText(

        frame,

        f"Stable frames: {len(prediction_history)}/{SMOOTHING_FRAMES}",

        (30, 143),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.40,

        (255, 255, 255),

        1
    )


    # ========================================================
    # BOTTOM MESSAGE
    # ========================================================

    cv2.putText(

        frame,

        "Q = Quit",

        (1180, 700),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

print("\nClosing camera...")

cap.release()

cv2.destroyAllWindows()

hand_landmarker.close()

print("Gesture recognition closed.")

