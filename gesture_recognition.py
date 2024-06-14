import cv2
import numpy as np

def detect_gesture(frame):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)  

   
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)  

   
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  
    gesture_labels = []

    
    height, _ = frame.shape[:2]


    top_threshold = height // 4 

    
    for contour in contours:
        
        area = cv2.contourArea(contour)

       
        if area > 1000:
          
            x, y, w, h = cv2.boundingRect(contour)

            
            if y > top_threshold:
                
                roi = frame[y:y+h, x:x+w]

                
                aspect_ratio = w / float(h)
                if aspect_ratio > 0.8:
                    
                    if detect_shaking_motion(frame, contour) and is_skin_color(roi):
                        label = "Hi"
                    else:
                        label = "Jami Rohan"
                else:
                    label = "Jami Rohan"

                
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                gesture_labels.append(label)

    
    cv2.imshow('Gesture Recognition', frame)

    return gesture_labels


def detect_shaking_motion(frame, contour):
    
    hull = cv2.convexHull(contour, returnPoints=False)

   
    defects = cv2.convexityDefects(contour, hull)

    
    if defects is not None:
        num_defects = defects.shape[0]
        if num_defects >= 8:  # Increased threshold for shaking motion detection
            return True
    return False

def is_skin_color(roi):
   
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

   
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    
    mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

    
    skin_percentage = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

    
    skin_threshold = 0.1  
    if skin_percentage > skin_threshold:
        return True
    else:
        return False

def main():
    
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            break

       
        gesture_labels = detect_gesture(frame)

        
        print("Detected Gestures:", gesture_labels)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
