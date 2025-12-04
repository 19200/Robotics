import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, PointStamped
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np
import cv2 
import pyrealsense2 as rs
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped


def quaternion_to_rpy(q: Quaternion):
    """
    Convert a quaternion into roll, pitch, and yaw (in radians).
    """
    x, y, z, w = q.x, q.y, q.z, q.w

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class ImageSubscriber(Node):
    
    def __init__(self):
        super().__init__('image_subscriber')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.subscription_image = self.create_subscription(
            Image,
            '/camera_depth/image_raw',
            self.image_callback,
            10)
        
        self.subscription_dimage = self.create_subscription(
            Image,
            '/camera_depth/depth/image_raw',
            self.dimage_callback,
            10)
        
        self.subscription_int = self.create_subscription(
            CameraInfo,
            '/camera_depth/camera_info',
            self.ins_callback,
            10)

        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
      
        self.ins = None
        self.image = None
        self.dimage = None
        self.Image_Publisher = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.go_to_points = []
        self.br = CvBridge()
        
        self.timer = self.create_timer(0.2, self.timer_callback)
    
        self.marker_counter = 0

    def odom_callback(self, msg):
        self.location = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        _, _, self.orientation = quaternion_to_rpy(msg.pose.pose.orientation)
    
    def ins_callback(self, data):
        self.ins = data
    
    def tf_from_cam_to_map(self):
        from_frame = 'camera_rgb_optical_frame'
        to_frame = 'map'
        
        now = rclpy.time.Time()
        
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame, now, timeout=rclpy.duration.Duration(seconds=1.0))
            return tf
        except Exception as e:
            self.get_logger().error(f"Failed to get transformation: {e}")
            return None
        
    def image_callback(self, data):
        #self.get_logger().info('Receiving video frame')
    
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        self.image = current_frame
        
    def dimage_callback(self, data):
        #self.get_logger().info('Receiving dvideo frame')
    
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        self.dimage = current_frame

    def object_colour_detect(self, lower_bound, upper_bound, current_frame):
        centroids = [] #from the rgb image
        depths = [] #from the depth image (pixel of centroid)
        
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Do some vision processing here to get the centroids of any green objects
        # `centroids` should contain a list pixel points (x,y) for each green centroid
        # `depths' should contain a list of the corresponding depth at those pixel points
        # Do some vision processing here to get the centroids of any green objects
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Erosion and Dilution to get rid of little green bits
        kernel = np.ones((15, 15), np.uint8)
        
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # advanced erosion
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # advanced dilution
        
        # Finding Midpoint
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in contours:

            area = cv2.contourArea(i)
            if area < 500:
                continue

            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                centroids.append([cx, cy])
                
                cv2.drawContours(current_frame, [i], -1, (0, 255, 0), 2)
                cv2.circle(current_frame, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(current_frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                h, w = self.dimage.shape[:2]
                if 0 <= cx < w and 0 <= cy < h:
                    depth_value = float(self.dimage[cy, cx])
                else:
                    depth_value = 0.0   # treat out-of-range as invalid depth

                if depth_value > 10.0:
                    depth_value = depth_value / 1000.0


                if depth_value <= 0 or depth_value > 3.0 or math.isnan(depth_value) or math.isinf(depth_value):

                    depths.append(None)
                else:
                    depths.append(depth_value)
        
        result = cv2.bitwise_and(current_frame, current_frame, mask=mask_clean)

        if not centroids or not depths:
            print("no centroids or depths")
            return None

        valid_centroids = []
        valid_depths = []

        for c, d in zip(centroids, depths):
            if d is not None and d > 0:
                valid_centroids.append(c)
                valid_depths.append(d)

        # Replace lists with filtered versions
        centroids = valid_centroids
        depths = valid_depths

        # If nothing left → no valid detections
        if len(centroids) == 0:
            print("No valid centroid-depth pairs → skipping frame")
            return None

        if self.ins is None:
            return None

        cameraInfo = self.ins

        _intrinsics = rs.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.k[2]
        _intrinsics.ppy = cameraInfo.k[5]
        _intrinsics.fx = cameraInfo.k[0]
        _intrinsics.fy = cameraInfo.k[4]
        _intrinsics.model = rs.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.d]

        points_3d = [rs.rs2_deproject_pixel_to_point(_intrinsics, centroids[x], depths[x]) for x in range(len(centroids))]
        
        point = PointStamped()
        point.header.frame_id = 'camera_rgb_optical_frame'   # <-- FIXED
        point.header.stamp = self.get_clock().now().to_msg()

        point.point.x = points_3d[0][0]
        point.point.y = points_3d[0][1]
        point.point.z = points_3d[0][2]

        X, Y, Z = points_3d[0]

        # Reject invalid numbers
        if any([math.isnan(X), math.isnan(Y), math.isnan(Z),
                math.isinf(X), math.isinf(Y), math.isinf(Z),
                Z <= 0]):
            print("Invalid 3D point (NaN/inf/zero depth) → skipping")
            return None

        tf = self.tf_from_cam_to_map()
        if tf is None:
            print("broken")
            return None

        point_world = do_transform_point(point, tf)
        return result, point_world

    def is_far_from_all_points(self, x, y, min_distance = 0.6): # tuning

        for px, py in self.go_to_points:
            dist = math.hypot(px - x, py - y)
            if dist < min_distance:
                return False   # too close = reject
        return True            # far from all = accept

    def timer_callback(self):

        if self.image is None or self.dimage is None:
            return

        # Make sure the images are valid NumPy arrays with data
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            return

        if not isinstance(self.dimage, np.ndarray) or self.dimage.size == 0:
            return
        
        current_frame = self.image
        
        if current_frame == []:
            return

        lower_green = np.array([45, 80, 30])
        upper_green = np.array([80, 255, 255])

        lower_red = np.array([0, 120, 80])
        upper_red = np.array([8, 255, 255])
        
        output_green = self.object_colour_detect(lower_green, upper_green, current_frame)

        if output_green is not None:
            result_g, point_world_g = output_green
            if point_world_g is not None and self.is_far_from_all_points(point_world_g.point.x,point_world_g.point.y):
                self.go_to_points.append((point_world_g.point.x,point_world_g.point.y))
                
                pose = PoseStamped()
                pose.header = point_world_g.header 

                pose.pose.position.x = point_world_g.point.x
                pose.pose.position.y = point_world_g.point.y
                pose.pose.position.z = 1.0

                self.Image_Publisher.publish(pose)
                cv2.imshow("Green Filtered Result", result_g)

        output_red = self.object_colour_detect(lower_red, upper_red, current_frame)

        if output_red is not None:
            result_r, point_world_r = output_red
            if point_world_r is not None and self.is_far_from_all_points(point_world_r.point.x,point_world_r.point.y):
                self.go_to_points.append((point_world_r.point.x,point_world_r.point.y))
                
                pose = PoseStamped()
                pose.header = point_world_r.header 

                pose.pose.position.x = point_world_r.point.x
                pose.pose.position.y = point_world_r.point.y
                pose.pose.position.z = 0.0

                self.Image_Publisher.publish(pose)
                cv2.imshow("Red Filtered Result", result_r)

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    