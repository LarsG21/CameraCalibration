import numpy as np
import cv2

# Making The Blank Image
image = np.zeros((1024,1024,3))
drawing = False
x = 0
y = 0
ix = 0
iy = 0
x_Start = 0
y_Start = 0
# Adding Function Attached To Mouse Callback
def draw(event,x,y,flags,params):
    global ix,iy,drawing,x_Start,y_Start
    # Left Mouse Button Down Pressed
    if(event==cv2.EVENT_LBUTTONDOWN):
        drawing = True
        x_Start = x
        y_Start = y
        print("Assigned Start Values", y_Start, x_Start)
    if(event==cv2.EVENT_MOUSEMOVE):
        if(drawing==True):
            #For Drawing Line
            #cv2.line(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
            ix = x
            iy = y
            # For Drawing Rectangle
    if(event==cv2.EVENT_LBUTTONUP):
        cv2.rectangle(image, pt1=(ix, iy), pt2=(x_Start, y_Start), color=(0, 0, 255), thickness=1)
        #print()
        drawing = False

def circle_shape(event,x,y,fags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,(x,y),50,(255,0,0),-1)


# Making Window For The Image
cv2.namedWindow("Window")

# Adding Mouse CallBack Event
cv2.setMouseCallback("Window",draw)

# Starting The Loop So Image Can Be Shown
while(True):

    cv2.imshow("Window",image)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()