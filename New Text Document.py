
import cv2 
import numpy 
 
def main():
#Captured the video file using the video capture , we can also use webcam for real time object tracking 


 
    # Capturing the video file using video capture
    video_file = cv2.VideoCapture('car-overhead-1.avi');

    #reading the estimated area size to be tracked
    templte_image = cv2.imread('temp.jpg');
    #converting the image to numpy array
    image_numpy = numpy.array(templte_image);
    # Background sub. Helps us to remove existing backgroung from the image which helps us to get the moving foreground easily from the background.

    #to write the video as output file 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #generate createBackgroundSubtractorMOG2 from last 700 frames 
    back_sub = cv2.createBackgroundSubtractorMOG2(history=750,varThreshold=20, detectShadows=True)
 
    #creating kerel and gen_kernel can be used to squeeze the dimensions 
    gen_kernel = numpy.ones((20,20),numpy.uint8)
    # Next we used numpy.ones() to squeez or modify the existing dimensions and return a new array 

    output_car = cv2.VideoWriter('output_car.avi', fourcc, 20.0, (300,300))
    while(True):
 
        #capturing the video's each display_frame
        return_vid, display_frame = video_file.read()
        
        # Tracked each frame of video 

 
        #deduce foreground mask using each display_frame
        foreground_mask = back_sub.apply(display_frame)
        # Deduced the foreground from the background 


        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, gen_kernel)
 
       #removing noise 
        foreground_mask = cv2.medianBlur(foreground_mask, 5) 
        # Reduce the noise 

        
        _, foreground_mask = cv2.threshold(foreground_mask,127,255,cv2.THRESH_BINARY)
 
        # Find the index of the largest contour and draw bounding box
        foreground_max_border = foreground_mask

        contours, hierarchy = cv2.findContours(foreground_max_border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        areas = [cv2.contourArea(cont_) for cont_ in contours]
        
 
        #<1 means no countours exsiting in the display_frame
        if len(areas) < 1:
            
            #Showing resultant display_frame
            cv2.imshow('display_frame',display_frame)
 
           
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break
 
        
            continue
 
        else:
            #detecting moving object inside the display_frame.
            

            
            if numpy.argmax(areas)>numpy.argmax(templte_image):

                highest_val_index = numpy.argmax(templte_image)
            else:
                highest_val_index = numpy.argmax(areas)
        #for the border around the mmoving object
        # Detected the index of largest moving object (contour) and enclosing it inside the bounding boxes .
            
        _contours = contours[highest_val_index]
        val_x,val_y,width,height  = cv2.boundingRect(_contours)
        cv2.rectangle(display_frame,(val_x,val_y),(val_x+width,val_y+height ),(0,0,2000),1)
 
        #for centre of the oject existing in the display_frame
        # Retrieve the centre of moving object and coordinates of its centre
        

        beta_x = val_x + int(width/2)

        beta_y = val_y + int(height /2)

        cv2.circle(display_frame,(beta_x,beta_y),4,(255,255,255),-1)
 
        #using put text to show the centre of object in side the display_frame
        text = "X: "       + str(beta_x) + ", Y: "           + str(beta_y)
        cv2.putText(display_frame, text, (beta_x - 10, beta_y - 10),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, (0, 255, 0), 2)

        if return_vid:
            vidio_output=cv2.resize(display_frame,(300,300))

            output_car.write(vidio_output)

        # showing frames 
        cv2.imshow('display_frame',display_frame)
 
        #exiting the loop after pressing key
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
 
    #releasing the output file , releasing the video file , destroy all windows at end of execution
    video_file.release()
    output_car.release()
    cv2.destroyAllWindows()
 
main()