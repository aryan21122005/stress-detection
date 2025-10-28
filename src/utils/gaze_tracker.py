import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional

class GazeTracker:
    """Tracks eye gaze and head pose to estimate screen attention."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Indices of facial landmarks for eyes and face
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left Mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ], dtype=np.float64)
        
    def _get_face_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect face landmarks using MediaPipe."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for idx, landmark in enumerate(landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmark_points.append((x, y))
            
        return {
            'all': landmark_points,
            'left_eye': [landmark_points[i] for i in self.LEFT_EYE],
            'right_eye': [landmark_points[i] for i in self.RIGHT_EYE],
            'face_oval': [landmark_points[i] for i in self.FACE_OVAL]
        }
    
    def _are_eyes_open(self, landmarks: list) -> bool:
        """Check if eyes are open using multiple criteria for robustness."""
        if not landmarks or len(landmarks) < 400:  # Need enough landmarks for accurate detection
            return False
            
        try:
            # Method 1: Eye aspect ratio (lower when eyes are closed)
            left_eye_ratio = self._get_eye_aspect_ratio([landmarks[i] for i in self.LEFT_EYE[:6]])
            right_eye_ratio = self._get_eye_aspect_ratio([landmarks[i] for i in self.RIGHT_EYE[:6]])
            
            # Method 2: Vertical distance between eyelids
            left_upper = landmarks[159]  # Upper eyelid
            left_lower = landmarks[145]  # Lower eyelid
            left_eye_open = left_lower[1] - left_upper[1] > 2  # Vertical distance threshold
            
            right_upper = landmarks[386]  # Upper eyelid
            right_lower = landmarks[374]  # Lower eyelid
            right_eye_open = right_lower[1] - right_upper[1] > 2  # Vertical distance threshold
            
            # Both methods must agree that eyes are open
            return ((left_eye_ratio > 0.2 and right_eye_ratio > 0.2) and 
                   (left_eye_open or right_eye_open))
                   
        except Exception as e:
            print(f"Eye detection error: {e}")
            return False
    
    def _get_head_pose(self, landmarks: Dict, frame_shape: Tuple[int, int]) -> Tuple[float, np.ndarray]:
        """Estimate head pose using solvePnP."""
        # 2D image points from landmarks
        image_points = np.array([
            landmarks['all'][1],    # Nose tip
            landmarks['all'][152],  # Chin
            landmarks['all'][33],   # Left eye left corner
            landmarks['all'][263],  # Right eye right corner
            landmarks['all'][61],   # Left mouth corner
            landmarks['all'][291]   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Solve PnP
        dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return 0.0, None
            
        # Project a 3D point for direction vector
        nose_end_point2D, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vec, 
            translation_vec, camera_matrix, dist_coeffs)
        
        # Calculate angle (simplified)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        # Calculate how centered the nose is
        frame_center = np.array([frame_shape[1]/2, frame_shape[0]/2])
        offset = np.linalg.norm(np.array(p1) - frame_center)
        max_offset = np.linalg.norm(frame_shape[:2])
        pose_score = 1.0 - min(offset / max_offset, 1.0)
        
        return pose_score, (p1, p2)
    
    def _get_eye_aspect_ratio(self, eye_points: list) -> float:
        """Calculate eye aspect ratio to detect if eyes are open."""
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        return ear
    
    def estimate_screen_attention(self, frame: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """
        Estimate how much the user is looking at the screen.
        Returns:
            float: Attention score from 0.0 (not looking) to 1.0 (directly facing)
            dict: Debug info with landmarks and visualization data
        """
        try:
            landmarks = self._get_face_landmarks(frame)
            if not landmarks:
                return 0.0, {'eyes_open': False}
            
            # Check if eyes are open (using more strict criteria)
            eyes_open = self._are_eyes_open(landmarks['all'])
            
            # If eyes are closed, return immediately with score 0
            if not eyes_open:
                return 0.0, {'eyes_open': False}
            
            # Get head pose
            pose_score, pose_points = self._get_head_pose(landmarks, frame.shape)
            
            # Calculate attention score (0.0 to 1.0)
            # More weight on head pose since it's more reliable
            attention_score = pose_score * 0.9 + 0.1  # Slight bias towards attention
            
            # Ensure score is within valid range
            attention_score = max(0.0, min(1.0, attention_score))
            
            debug_info = {
                'eyes_open': True,
                'pose_score': pose_score,
                'pose_points': None  # Don't return points to avoid drawing line
            }
            
            return attention_score, debug_info
            
        except Exception as e:
            print(f"Attention estimation error: {e}")
            return 0.0, {'eyes_open': False}
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()

# Singleton instance for easy access
gaze_tracker = GazeTracker()
