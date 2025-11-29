import cv2
import mediapipe as mp

# Global Setup for MediaPipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# --- Module 1: Connect to Camera ---
def start_camera(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print("Camera Connected!")
        return cap
    else:
        print("Error: Cannot connect to camera.")
        return None

# --- Module 2: Setup Video Saver ---
def setup_video_saver(cap, file_name):
    # Get width and height from camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (width, height))
    return out

# --- Module 3: Detect and Draw Faces ---
def detect_face(frame):
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process detection
    results = detector.process(img_rgb)
    
    # Draw boxes if faces found
    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(frame, detection)
            
    return frame

# --- Main Program ---
def main():
    # Replace with your IP
    ip_url = "http://192.168.1.54:8080/video"
    
    # 1. Start Camera
    camera = start_camera(ip_url)
    if camera is None: return

    # 2. Setup Recording
    recorder = setup_video_saver(camera, "my_video.mp4")

    print("Press 'q' to stop...")

    while True:
        # Read frame
        ret, frame = camera.read()
        if not ret: break

        # 3. Apply Face Detection
        frame = detect_face(frame)

        # 4. Save and Show
        recorder.write(frame)
        
        # Resize for display only
        small_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("My Project", small_frame)

        # Quit logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.release()
    recorder.release()
    cv2.destroyAllWindows()

if __name__ == "main":
    main()