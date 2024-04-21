import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
from ICA import *
import matplotlib.pyplot as plt
from Heart_Rate import *
from Heart_Rate import _calculate_peak_hr, _calculate_fft_hr
from scipy.signal import find_peaks
from CHROME_DEHAAN import *
from LGI import *
from POS_WANG import *
from post_processing import _detrend
from read_GT import *
frame_size = (75, 75)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)


face_frames = []
no_frames_batch = 0 # To Prepare of the one batch to input to rPPG prediction Algorithm
itr = 0
hr_mv_chrome = 0
hr_fft_chrome = 0
hr_fft_lgi = 0
hr_mv_ica = 0
hr_fft_ica = 0
hr_fft_pos = 0
no_of_batch = 1
hr_fft_gt = 0
list_hr = 0


ground_truth = False

# Open a video capture object

if ground_truth:
    cap = cv2.VideoCapture('vid.avi')
    labels = read_ground_truth('ground_truth.txt')
    
    
else:
    cap = cv2.VideoCapture(1)
    

while cap.isOpened():
    ret, frame = cap.read()
    
    
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Perform face detection
    #--------------------------------------------------------------------------#
    frame, bboxs = detector.findFaces(frame, draw=True)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            # center = bbox["center"]
            x, y, w, h = bbox['bbox']
            # score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
            cvzone.putTextRect(frame, f'hr_fft_ica:{int(hr_fft_ica)}', (20, 50), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_mv_ica:{int(np.average(hr_mv_ica))}', (20, 100), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_fft_chrome:{int(hr_fft_chrome)}', (20, 150), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_fft_lgi:{int(hr_fft_lgi)}', (20, 200), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_fft_pos:{int(hr_fft_pos)}', (370, 50), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_fft_gt:{int(hr_fft_gt)}', (370, 100), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(frame, f'hr_avr:{int(np.average(list_hr))}', (370, 150), scale=2, colorR=(0, 0, 255))
            
            # cvzone.cornerRect(frame, (x, y, w, h))
            face_frame = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_frame, frame_size)
            # skin, _ = segment_skin(face_frame)
            #     # cv2.imshow('abdo', skin)
            
            # face_frame_averaged = spatial_average(skin)
            # face_frame_averaged = spatial_average(face_frame)
            
            face_frames.append(resized_face)
           
            no_frames_batch += 1
            #--------rPPG Detection------
            #Update Each 30 Frames
            if(no_frames_batch > 200):
                print(f'batch_{no_of_batch}')
                
                #----------Estimate Heart_Rate From Ground_Truth--------
                if ground_truth:
                    hr_fft_gt = _calculate_fft_hr(labels[itr:200+itr], fs=30)
                #--------------------------------------------------------
                
                face_frames_arr = np.array(face_frames)
                
                # ----------------------ICA Algorithm----------------------
                rPPG_ica = ICA_POH(face_frames_arr[itr:200+itr], 30)
                rPPG_ica = _detrend(rPPG_ica, 100)
                
                # bandpass filter between [0.75, 2.5] Hz
                # equals [45, 150] beats per min
                [b, a] = butter(1, [0.75 / 30 * 2, 2.5 / 30 * 2], btype='bandpass')
                rPPG_ica = scipy.signal.filtfilt(b, a, np.double(rPPG_ica))
                
                # hr_peak = _calculate_peak_hr(rPPG, 30)

                x_peaks_ica ,_ = find_peaks(rPPG_ica, height=0.25)

                hr_mv_ica = moving_heart_rate(x_peaks_ica, 3, 30)
                
                hr_fft_ica = _calculate_fft_hr(rPPG_ica, fs=30)
                
                
                #############################################################
                
                # ----------------------CHROME Algorithm----------------------
                rPPG_chrome = CHROME_DEHAAN(face_frames_arr[0+itr:200+itr], 30)
                rPPG_chrome = _detrend(rPPG_chrome, 100)
                # hr_peak = _calculate_peak_hr(rPPG, 30)
                
                [b, a] = butter(1, [0.75 / 30 * 2, 2.5 / 30 * 2], btype='bandpass')
                rPPG_chrome = scipy.signal.filtfilt(b, a, np.double(rPPG_chrome))

                x_peaks_chrome ,_ = find_peaks(rPPG_chrome, height=0.25)

                hr_mv_chrome = moving_heart_rate(x_peaks_chrome, 3, 30)
                hr_mv_chrome = np.average(hr_mv_chrome)
                
                hr_fft_chrome = _calculate_fft_hr(rPPG_chrome, fs=30)

                #############################################################
                
                # ----------------------ICA Algorithm----------------------
                rPPG_lgi = LGI(face_frames_arr[0+itr:200+itr])
                rPPG_lgi = _detrend(rPPG_lgi, 100)
                
                [b, a] = butter(1, [0.75 / 30 * 2, 2.5 / 30 * 2], btype='bandpass')
                rPPG_lgi = scipy.signal.filtfilt(b, a, np.double(rPPG_lgi))
                
                # hr_peak = _calculate_peak_hr(rPPG, 30)

                # x_peaks_ica ,_ = find_peaks(rPPG_ica, height=0.25)

                # hr_mv_ica = moving_heart_rate(x_peaks_ica, 3, 30)
                
                hr_fft_lgi = _calculate_fft_hr(rPPG_lgi, fs=30)
                
                #############################################################
                # ----------------------POS Algorithm----------------------
                rPPG_pos = POS_WANG(face_frames_arr[0+itr:200+itr], 30)
                rPPG_pos = _detrend(rPPG_pos, 100)
                
                [b, a] = butter(1, [0.75 / 30 * 2, 2.5 / 30 * 2], btype='bandpass')
                rPPG_pos = scipy.signal.filtfilt(b, a, np.double(rPPG_pos))
                
                # hr_peak = _calculate_peak_hr(rPPG, 30)

                # x_peaks_chrome ,_ = find_peaks(rPPG_chrome, height=0.25)

                # hr_mv_chrome = moving_heart_rate(x_peaks_chrome, 3, 30)
                # hr_mv_chrome = np.average(hr_mv_chrome)
                
                hr_fft_pos = _calculate_fft_hr(rPPG_pos, fs=30)

                #############################################################
                print(f"hr_fft_ica:{int(hr_fft_ica)}")
                print(f"hr_mv_ica:{int(np.average(hr_mv_ica))}")
                print(f"hr_fft_chrome:{int(hr_fft_chrome)}")
                print(f"hr_fft_lgi:{hr_fft_lgi}")
                print(f"hr_fft_pos:{hr_fft_pos}")
                list_hr = [int(hr_fft_ica), hr_fft_chrome, hr_fft_pos]
                print(f"hr_avr:{int(np.average(list_hr))}")
                
                no_frames_batch = 0
                itr += 200
                
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    # Display the image in a window named 'Image'
    cv2.imshow("Image", frame)
    #--------------------------------------------------------------------------#

    # Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

# face_frames = np.array(face_frames)

# print(face_frames.shape)

# rPPG = ICA_POH(face_frames, 30)

# hr_peak = _calculate_peak_hr(rPPG, 30)

# x_peaks ,_ = find_peaks(rPPG, height=0.25)

# hr_mv = moving_heart_rate(x_peaks, 2, 30)

# hr_fft = _calculate_fft_hr(rPPG, fs=30)

# print(f"hr_peak:{hr_peak}")
# print(f"hr_mv:{np.average(hr_mv)}")
# print(f"hr_fft:{hr_fft}")

# plt.plot(rPPG_chrome)

# plt.show()



#--------------display frames--------------
# # Create a window to display frames
# cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)

# # Loop through each frame and display it
# for frame in face_frames:
#     cv2.imshow('Frames', frame)
    
#     # Press 'q' to exit the window
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Release the window and close it
# cv2.destroyAllWindows()

#------------------------------------------------------