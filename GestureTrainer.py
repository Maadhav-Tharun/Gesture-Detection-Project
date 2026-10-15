import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import joblib
import time
import threading
import queue
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "gesture_dataset.csv"
MODEL_FILE = "gesture_model.pkl"

HAND_MODEL_FILE = "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

# Increased from 100 to 150
SAMPLES_PER_GESTURE = 150

# Capture delay
CAPTURE_DELAY = 0.08

# Number of artificial samples generated from each real sample
AUGMENTATION_FACTOR = 4


# ============================================================
# DROIDCAM CAMERA SETTINGS
# ============================================================

CAMERA_INDEXES_TO_TEST = [1, 2, 3, 4, 5, 0]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

current_label = None

collecting = False

sample_count = 0

last_capture_time = 0

program_running = True

command_queue = queue.Queue()


# ============================================================
# TERMINAL COMMAND THREAD
# ============================================================

def terminal_input():

    print("\n")
    print("=" * 60)
    print("          IMPROVED GESTURE AI TRAINER")
    print("=" * 60)

    print("LABEL A")
    print("LABEL B")
    print("LABEL HELLO")
    print("START")
    print("STOP")
    print("STATUS")
    print("TRAIN")
    print("QUIT")

    print("=" * 60)

    while program_running:

        try:

            command = input("\nCOMMAND: ").strip()

            if command:

                command_queue.put(command)

        except:

            break


threading.Thread(
    target=terminal_input,
    daemon=True
).start()


# ============================================================
# DOWNLOAD HAND MODEL
# ============================================================

if not os.path.exists(HAND_MODEL_FILE):

    print("\nDownloading MediaPipe hand model...")

    try:

        urllib.request.urlretrieve(
            HAND_MODEL_URL,
            HAND_MODEL_FILE
        )

        print(
            "Hand model downloaded successfully!"
        )

    except Exception as error:

        print(
            "Download error:",
            error
        )

        exit()


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print(
    "\nLoading MediaPipe Hand Landmarker..."
)

try:

    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL_FILE
    )

    options = vision.HandLandmarkerOptions(

        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        # TWO HANDS
        num_hands=2,

        min_hand_detection_confidence=0.60,

        min_hand_presence_confidence=0.60,

        min_tracking_confidence=0.60
    )

    hand_landmarker = (
        vision.HandLandmarker.create_from_options(
            options
        )
    )

    print(
        "MediaPipe loaded successfully!"
    )

except Exception as error:

    print(
        "\nMediaPipe error:"
    )

    print(error)

    exit()


# ============================================================
# CREATE NEW DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):

    columns = []

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    for i in range(21):

        columns.append(
            f"left_x{i}"
        )

        columns.append(
            f"left_y{i}"
        )

        columns.append(
            f"left_z{i}"
        )

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    for i in range(21):

        columns.append(
            f"right_x{i}"
        )

        columns.append(
            f"right_y{i}"
        )

        columns.append(
            f"right_z{i}"
        )

    columns.append("label")

    pd.DataFrame(
        columns=columns
    ).to_csv(
        DATASET_FILE,
        index=False
    )

    print(
        "\nNew normalized dataset created!"
    )


# ============================================================
# IMPROVED HAND FEATURE EXTRACTION
# ============================================================

def extract_hand(hand_landmarks):

    """
    Creates 63 normalized features.

    Improvements:
    1. Wrist becomes origin.
    2. Hand size is normalized.
    3. Small changes in distance from camera
       have less effect.
    """

    wrist = hand_landmarks[0]

    raw_features = []

    # --------------------------------------------------------
    # STEP 1
    # Convert landmarks to wrist-relative coordinates
    # --------------------------------------------------------

    for landmark in hand_landmarks:

        x = landmark.x - wrist.x

        y = landmark.y - wrist.y

        z = landmark.z - wrist.z

        raw_features.append(x)
        raw_features.append(y)
        raw_features.append(z)


    # --------------------------------------------------------
    # STEP 2
    # Calculate hand size
    # --------------------------------------------------------

    distances = []

    for i in range(1, 21):

        x = hand_landmarks[i].x - wrist.x

        y = hand_landmarks[i].y - wrist.y

        z = hand_landmarks[i].z - wrist.z

        distance = np.sqrt(
            x * x +
            y * y +
            z * z
        )

        distances.append(distance)


    # --------------------------------------------------------
    # STEP 3
    # Find scale
    # --------------------------------------------------------

    scale = max(
        distances
    )

    # Avoid division by zero

    if scale < 0.0001:

        scale = 1.0


    # --------------------------------------------------------
    # STEP 4
    # Normalize all coordinates
    # --------------------------------------------------------

    normalized_features = []

    for value in raw_features:

        normalized_features.append(
            value / scale
        )


    return normalized_features


# ============================================================
# TWO-HAND FEATURE EXTRACTION
# ============================================================

def extract_two_hands(result):

    # 63 LEFT
    # 63 RIGHT
    # TOTAL = 126

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
# DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Little
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (0, 17)
    ]


    # --------------------------------------------------------
    # LINES
    # --------------------------------------------------------

    for start, end in connections:

        p1 = landmarks[start]

        p2 = landmarks[end]

        x1 = int(
            p1.x * width
        )

        y1 = int(
            p1.y * height
        )

        x2 = int(
            p2.x * width
        )

        y2 = int(
            p2.y * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # POINTS
    # --------------------------------------------------------

    for point in landmarks:

        x = int(
            point.x * width
        )

        y = int(
            point.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# ============================================================
# SAVE SAMPLE
# ============================================================

def save_sample(
    features,
    label
):

    row = features + [label]

    pd.DataFrame(
        [row]
    ).to_csv(
        DATASET_FILE,
        mode="a",
        header=False,
        index=False
    )


# ============================================================
# DATASET STATUS
# ============================================================

def show_status():

    try:

        data = pd.read_csv(
            DATASET_FILE
        )

    except Exception as error:

        print(
            "Dataset error:",
            error
        )

        return


    print("\n")
    print("=" * 50)
    print("DATASET STATUS")
    print("=" * 50)

    print(
        "Total samples:",
        len(data)
    )

    if len(data) > 0:

        print("\nSamples per gesture:")

        print(
            data["label"].value_counts()
        )

    else:

        print(
            "No samples collected."
        )

    print("=" * 50)


# ============================================================
# TRAIN IMPROVED MODEL
# ============================================================

def train_model():

    print("\n")
    print("=" * 60)
    print("        IMPROVED MODEL TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        data = pd.read_csv(
            DATASET_FILE
        )

    except Exception as error:

        print(
            "Dataset error:",
            error
        )

        return


    if len(data) == 0:

        print(
            "ERROR: Dataset is empty!"
        )

        return


    # --------------------------------------------------------
    # DISPLAY COUNTS
    # --------------------------------------------------------

    print("\nReal samples:")

    print(
        data["label"].value_counts()
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = data.drop(
        "label",
        axis=1
    ).values.astype(
        np.float32
    )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    y = data["label"].values


    # --------------------------------------------------------
    # CHECK GESTURES
    # --------------------------------------------------------

    unique_labels = np.unique(y)

    if len(unique_labels) < 2:

        print(
            "\nERROR:"
        )

        print(
            "Train at least TWO different gestures."
        )

        return


    # ========================================================
    # DATA AUGMENTATION
    # ========================================================

    print(
        "\nCreating augmented training samples..."
    )

    augmented_X = [X]

    augmented_y = [y]


    for augmentation_number in range(
        AUGMENTATION_FACTOR
    ):

        # ----------------------------------------------------
        # Small landmark noise
        # ----------------------------------------------------

        noise = np.random.normal(
            loc=0.0,
            scale=0.008,
            size=X.shape
        )


        # ----------------------------------------------------
        # Don't add noise to zero missing-hand areas
        # ----------------------------------------------------

        augmented = X.copy()

        nonzero_mask = (
            np.abs(X) > 0.0001
        )

        augmented[
            nonzero_mask
        ] += noise[
            nonzero_mask
        ]


        # ----------------------------------------------------
        # Small global variation
        # ----------------------------------------------------

        scale_variation = np.random.uniform(
            0.98,
            1.02,
            size=(X.shape[0], 1)
        )

        augmented = (
            augmented *
            scale_variation
        )


        augmented_X.append(
            augmented
        )

        augmented_y.append(
            y.copy()
        )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    X_final = np.vstack(
        augmented_X
    )

    y_final = np.concatenate(
        augmented_y
    )


    print(
        "\nOriginal samples:",
        len(X)
    )

    print(
        "Training samples after augmentation:",
        len(X_final)
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X_final,

            y_final,

            test_size=0.20,

            random_state=42,

            stratify=y_final
        )
    )


    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nCreating improved Random Forest..."
    )


    model = RandomForestClassifier(

        # Increased from 300
        n_estimators=600,

        # More robust trees
        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        class_weight="balanced",

        bootstrap=True,

        random_state=42,

        n_jobs=-1
    )


    print(
        "Training..."
    )


    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # TEST
    # ========================================================

    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    # ========================================================
    # SAVE
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("          TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"\nModel saved as:"
    )

    print(
        MODEL_FILE
    )


    print(
        "\nClasses:"
    )

    for label in model.classes_:

        print(
            "   ",
            label
        )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            prediction,
            zero_division=0
        )
    )


    print("=" * 60)


# ============================================================
# PROCESS COMMAND
# ============================================================

def process_command(command):

    global current_label

    global collecting

    global sample_count

    global last_capture_time

    global program_running


    command = command.strip()

    upper = command.upper()


    # ========================================================
    # QUIT
    # ========================================================

    if upper == "QUIT":

        collecting = False

        program_running = False

        print(
            "\nClosing trainer..."
        )

        return


    # ========================================================
    # LABEL
    # ========================================================

    if upper.startswith("LABEL "):

        label = command[6:].strip()

        if not label:

            print(
                "Please enter a label."
            )

            return


        current_label = label.upper()

        sample_count = 0

        collecting = False


        print("\n")
        print("=" * 50)

        print(
            f"SELECTED GESTURE: {current_label}"
        )

        print("=" * 50)

        print(
            "Type START to begin collecting."
        )

        return


    # ========================================================
    # START
    # ========================================================

    if upper == "START":

        if current_label is None:

            print(
                "\nERROR:"
            )

            print(
                "First type LABEL A"
            )

            return


        sample_count = 0

        collecting = True

        last_capture_time = 0


        print("\n")
        print("=" * 60)

        print(
            f"COLLECTING GESTURE: {current_label}"
        )

        print("=" * 60)

        print(
            "KEEP THE SAME GESTURE!"
        )

        print(
            "Small hand-position changes are GOOD."
        )

        print(
            "Small wrist rotations are GOOD."
        )

        print(
            "Slightly closer/farther is GOOD."
        )

        print(
            "DO NOT change the actual sign."
        )

        print("=" * 60)

        return


    # ========================================================
    # STOP
    # ========================================================

    if upper == "STOP":

        collecting = False

        print(
            "\nCollection stopped."
        )

        return


    # ========================================================
    # STATUS
    # ========================================================

    if upper == "STATUS":

        show_status()

        return


    # ========================================================
    # TRAIN
    # ========================================================

    if upper == "TRAIN":

        collecting = False

        train_model()

        return


    # ========================================================
    # UNKNOWN
    # ========================================================

    print(
        "\nUnknown command."
    )

    print(
        "Use LABEL, START, STOP, STATUS, TRAIN or QUIT."
    )


# ============================================================
# DROIDCAM SETUP
# ============================================================

print("\n")
print("=" * 60)
print("             DROIDCAM USB SETUP")
print("=" * 60)

print("""
Phone:

1. Connect USB cable.
2. Enable USB Debugging.
3. Open DroidCam.
4. Select USB mode.
5. Select 2310EPCA4T.
6. Video = ON.
7. Click START.
8. Make sure phone camera is visible.
""")


input(
    "\nPress ENTER after DroidCam has started..."
)


# ============================================================
# FIND DROIDCAM
# ============================================================

print(
    "\nSearching for DroidCam..."
)

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


    success, test_frame = (
        test_cap.read()
    )


    if (
        success
        and
        test_frame is not None
    ):

        print(
            f"Camera {index} available."
        )


        if index != 0:

            cap = test_cap

            selected_camera = index

            break


    test_cap.release()


# ============================================================
# CAMERA NOT FOUND
# ============================================================

if cap is None:

    print("\n")
    print("=" * 60)
    print("          DROIDCAM NOT FOUND")
    print("=" * 60)

    print("""
Make sure:

1. DroidCam is running.
2. USB mode is selected.
3. 2310EPCA4T is selected.
4. Video is enabled.
5. Start is pressed.
6. Phone camera is visible in DroidCam.
""")

    hand_landmarker.close()

    program_running = False

    exit()


# ============================================================
# CAMERA SUCCESS
# ============================================================

print("\n")
print("=" * 60)

print(
    "DROIDCAM FOUND!"
)

print(
    f"Camera index: {selected_camera}"
)

print("=" * 60)


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

WINDOW_NAME = (
    "Improved Gesture AI Trainer - DroidCam"
)


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
# MAIN LOOP
# ============================================================

while program_running:

    # ========================================================
    # COMMANDS
    # ========================================================

    while not command_queue.empty():

        command = (
            command_queue.get()
        )

        process_command(
            command
        )


    # ========================================================
    # READ CAMERA
    # ========================================================

    success, frame = cap.read()


    if not success:

        print(
            "DroidCam frame error!"
        )

        break


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = int(
        time.monotonic() * 1000
    )


    if timestamp <= last_timestamp:

        timestamp = (
            last_timestamp + 1
        )


    last_timestamp = timestamp


    # ========================================================
    # DETECT HANDS
    # ========================================================

    try:

        result = (
            hand_landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )

    except Exception as error:

        print(
            "MediaPipe error:",
            error
        )

        continue


    # ========================================================
    # HAND COUNT
    # ========================================================

    hands_detected = len(
        result.hand_landmarks
    )


    # ========================================================
    # DRAW HANDS
    # ========================================================

    for landmarks in result.hand_landmarks:

        draw_hand(
            frame,
            landmarks
        )


    # ========================================================
    # COLLECTION
    # ========================================================

    if collecting:

        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if hands_detected == 0:

            cv2.putText(
                frame,
                "SHOW YOUR HAND",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        else:

            current_time = time.time()


            if (
                current_time
                -
                last_capture_time
                >=
                CAPTURE_DELAY
            ):


                # ------------------------------------------------
                # EXTRACT NORMALIZED FEATURES
                # ------------------------------------------------

                features = (
                    extract_two_hands(
                        result
                    )
                )


                # ------------------------------------------------
                # VERIFY 126 FEATURES
                # ------------------------------------------------

                if len(features) == 126:

                    save_sample(
                        features,
                        current_label
                    )

                    sample_count += 1

                    last_capture_time = (
                        current_time
                    )


                    print(
                        f"{current_label}: "
                        f"{sample_count}/"
                        f"{SAMPLES_PER_GESTURE}"
                    )


                # ------------------------------------------------
                # COMPLETED
                # ------------------------------------------------

                if (
                    sample_count
                    >=
                    SAMPLES_PER_GESTURE
                ):

                    collecting = False


                    print("\n")
                    print("=" * 50)

                    print(
                        f"{current_label} COMPLETED!"
                    )

                    print(
                        f"{SAMPLES_PER_GESTURE} "
                        "samples collected."
                    )

                    print("=" * 50)

                    print(
                        "Type LABEL <next gesture>"
                    )

                    print(
                        "then START"
                    )


    # ========================================================
    # COMPACT PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (15, 15),
        (390, 185),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # CURRENT GESTURE
    # ========================================================

    gesture_text = (
        current_label
        if current_label
        else "NONE"
    )


    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # STATUS
    # ========================================================

    status = (
        "COLLECTING"
        if collecting
        else "WAITING"
    )


    status_color = (
        (0, 255, 0)
        if collecting
        else (255, 255, 255)
    )


    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )


    # ========================================================
    # HAND COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Hands: {hands_detected}",
        (30, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )


    # ========================================================
    # SAMPLE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{SAMPLES_PER_GESTURE}",
        (150, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # TRAINING INSTRUCTION
    # ========================================================

    if collecting:

        cv2.putText(
            frame,
            "SAME SIGN",
            (30, 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Move hand slightly",
            (30, 157),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1
        )

    else:

        cv2.putText(
            frame,
            "Waiting for START",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    progress = min(
        sample_count /
        SAMPLES_PER_GESTURE,
        1.0
    )


    height, width = (
        frame.shape[:2]
    )


    bar_width = 500

    bar_height = 14

    bar_x = 30

    bar_y = height - 45


    # Background

    cv2.rectangle(
        frame,
        (
            bar_x,
            bar_y
        ),
        (
            bar_x + bar_width,
            bar_y + bar_height
        ),
        (80, 80, 80),
        -1
    )


    # Progress

    filled = int(
        bar_width *
        progress
    )


    cv2.rectangle(
        frame,
        (
            bar_x,
            bar_y
        ),
        (
            bar_x + filled,
            bar_y + bar_height
        ),
        (0, 255, 0),
        -1
    )


    # ========================================================
    # CAMERA INFO
    # ========================================================

    cv2.putText(
        frame,
        f"DroidCam | {selected_camera}",
        (
            width - 240,
            height - 25
        ),
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

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        print(
            "\nQ pressed."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

program_running = False

collecting = False

cap.release()

cv2.destroyAllWindows()

hand_landmarker.close()


print("\n")
print("=" * 60)

print(
    "GESTURE TRAINER CLOSED"
)

print("=" * 60)
