import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class LaneTracker:
    def __init__(self):
        # Sliding window parameters
        self.nwindows = 9
        self.margin = 50      # Width of the windows +/- margin
        self.minpix = 50      # Min pixels found to recenter window
        
    def get_binary_image(self, img):
        """Applies color and edge thresholding to find lane pixels."""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # Thresholding the S channel (saturation) works well for lanes
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > 100) & (s_channel <= 255)] = 1
        return binary_output

    def find_lane_pixels(self, binary_warped):
        """Performs the sliding window search."""
        # Create an output image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = int(binary_warped.shape[0] // self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated later for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds] 
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

class CamViewerNode(Node):
    def __init__(self):
        super().__init__('cam_viewer')
        self.bridge = CvBridge()
        self.tracker = LaneTracker()
        self.image = None

        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.img_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(0.03, self.process_images)

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def process_images(self):
        if self.image is None:
            return
            
        # 1. Resize and Get Binary
        h, w = 240, 320
        f = cv2.resize(self.image, (w, h))
        binary = self.tracker.get_binary_image(f)

        # 2. Track Lanes (Coordinates)
        lx, ly, rx, ry, debug_img = self.tracker.find_lane_pixels(binary)

        # 3. Fit a polynomial if pixels were found
        if len(lx) > 0 and len(rx) > 0:
            left_fit = np.polyfit(ly, lx, 2)
            right_fit = np.polyfit(ry, rx, 2)
            # You can now use these coefficients to calculate curvature or offset
        
        cv2.imshow("Sliding Windows", debug_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CamViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()