import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
class NavigateToPoseActionClient(Node):
    MOVE,SCAN,OBJECT,AVOID,DONE,TURNTO = range(6)
    def __init__(self):
        super().__init__('NavigateToPose_action_client')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info("Waiting for Nav2 action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 not available! Navigation will not work.")
        else:
            self.get_logger().info("Nav2 action server is ready!")
        self.publisher_twist = self.create_publisher(Twist, 'cmd_vel', 10)

        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_received_callback,
            10
        )

        self.subscription = self.create_subscription(#check later ros convetion
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        self.frontier_point = self.create_subscription(#check later ros convetion
            PointStamped,
            '/frontier_point',
            self.frontier_callback,
            10
        )
        
        self.state = self.SCAN
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.timer_period_stuck = 10.0
        self.prev_pos = None # seconds
        self.stuck_distance_threshold = 0.2 
        self.timer_stuck = self.create_timer(self.timer_period_stuck, self.timer_callback_stuck)
        self.ranges = []
        self.subscription
        self.scan_points = [] # could be, maybe, idk chat (2.4, 0.0), (1.41, -3.41), (-0.47, -0.57), (-1.59, -3.03), (-1.58, 2.98), (1.01, 3.99),
                            #(3.68, 2.01), (5.51, 3.01), (5.41, -2.07)
        self.go_to_points = []
        self.points_found = []# list of points found 
        self.is_navigating = False
        self.current_goal_handle = None
        self.turn_spd = math.pi / 5.0
        self.turn_counter = 0.0
        self.distance_remaining = None
        self.min_distance_to_object = 0.4
        self.marker_pub = self.create_publisher(Marker, "/eye_target_marker", 10)
        self.Robot_point = self.create_publisher(PointStamped, '/robotpoint', 10)
        self.marker_counter = 0
        self.min_wall_distance = 0.3
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_pose = None 
        
    def frontier_callback(self, msg): # Not being called
        x = msg.point.x
        y = msg.point.y
        
        self.get_logger().info("check21")
        # Add as a tuple to the list
        self.scan_points.append((x, y))

    def quaternion_to_rpy(self,q: Quaternion):
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

    def timer_callback_stuck(self):
        # Get robot position

        if self.state == self.SCAN or self.state == self.TURNTO: return
        x, y, yaw = self.get_robot_pose()

        if x is None:  # failed to get pose
            return

        if self.prev_pos is None:
            # First time storing pose
            self.prev_pos = (x, y)
            return

        px, py = self.prev_pos

        # Euclidean distance moved since last check
        dist = np.hypot(x - px, y - py)

        # Debug print
        self.get_logger().info(f"[STUCK CHECK] Moved: {dist:.3f} m")

        # If robot moved less than threshold → it's stuck
        if dist < self.stuck_distance_threshold:
            self.get_logger().warn("Robot appears STUCK → Canceling current goal!")

            try:
                if self.state == self.MOVE:
                    # cancel the goal (assuming you have nav2 action client named self.nav_client)
                    self.cancel_navigation()
                    if self.scan_points:
                        self.is_navigating = False
                        self.scan_points.pop(0)
                elif self.state == self.OBJECT:
                    self.cancel_navigation()
                    self.is_navigating = False
                    self.state = self.TURNTO
            except Exception as e:
                self.get_logger().error(f"Failed to cancel goal: {e}")

        # Update for next cycle
        self.prev_pos = (x, y)


    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except TransformException:
            # Transform not ready yet
            return None
    
    def send_goala(self, x, y, yaw = 0.0):
        self.get_logger().info("ARE WE HANGING :-)")
        
        self._action_client.wait_for_server()

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.w = 1.0  # Neutral orientation (no rotation)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0

        self.get_logger().info("CHECK 54 :-)")
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = ''
        self.get_logger().info('point %f %f' % (x, y))
        self.publish_marker(x,y,0.0)
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        return send_goal_future

    def cancel_navigation(self):
        if self.current_goal_handle is None:
            self.get_logger().info("No goal to cancel.")
            return

        self.get_logger().warn("Cancelling current goal…")
        cancel_future = self.current_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def calculate_angles(self, opposite, adjecent):
        return math.degrees(math.atan2(opposite, adjecent))

    def can_move_foward(self, opposite, adjacent):
        scan_angle = int(self.calculate_angles(opposite, adjacent))

        
        for x in range(-scan_angle, scan_angle + 1):
            idx = x % 360
            distance = self.conv_angle(idx)

            forward_component = math.cos(math.radians(x)) * distance

            if forward_component < self.min_wall_distance:
                return False

        return True

    def conv_angle(self,angle):
        if len(self.ranges) == 0:
            return 0.01
        else:
            return self.ranges[math.floor(angle / (360/len(self.ranges)))]
    def cancel_done_callback(self, future):
        try:
            cancel_result = future.result()

            if len(cancel_result.goals_canceling) > 0:
                self.get_logger().warn("Goal successfully cancelled.")
            else:
                self.get_logger().warn("Goal cancel request was received, but no goals are canceling.")

        except Exception as e:
            self.get_logger().error(f"Error while cancelling goal: {e}")


        self.is_navigating = False
        
        self.current_goal_handle = None

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Goal was rejected by the server.")
            self.is_navigating = False
            return

        self.get_logger().info("Goal accepted :)")
        self.current_goal_handle = goal_handle
        self.is_navigating = True

        # Request the result asynchronously
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result  # Proper extraction of result object

        if result.result == 0:
            if self.state == self.MOVE:
                self.state = self.SCAN
                self.get_logger().warn("Move Finished Scanning")
            if self.state == self.OBJECT:
                self.state = self.MOVE
                self.get_logger().warn("object found publishing")
                new_point = self.go_to_points[0]
                self.publish_marker(new_point[0], new_point[1], new_point[2])
                self.points_found.append(self.go_to_points.pop(0))
        elif result.result == 1:
            self.get_logger().warn("Navigation cancelled")
        elif result.result == 2:
            self.get_logger().error("Navigation FAILED — goal unreachable or planning failed!")

            # Handle failure based on state
            if self.state == self.MOVE:
                self.is_navigating = False
                if self.scan_points:
                    self.scan_points.pop(0)

            elif self.state == self.OBJECT:
                self.is_navigating = False
                if self.go_to_points:
                    new_point = self.go_to_points[0]
                    self.publish_marker(new_point[0], new_point[1], new_point[2])
                    self.points_found.append(self.go_to_points.pop(0))

        # Reset navigation flags
        self.is_navigating = False
        self.current_goal_handle = None

        self.get_logger().info("Finished :)")


    def target_received_callback(self, pose_msg):
       
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z

        self.go_to_points.append((x, y, z))## adds target to list

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.distance_remaining = feedback_msg.feedback.distance_remaining
        ##self.get_logger().info('Received feedback: ')

    def publish_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "eye_targets"
        self.marker_counter += 1
        marker.id = self.marker_counter
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05  # Just above ground for visibility

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        if z == 0.0:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        print("publishing")
        self.marker_pub.publish(marker)


    def avoid_turn(self,msg):
        if (self.can_move_foward(0.2, 0.3)):
   
            self.stop(msg)
            if len(self.go_to_points) != 0:
                self.state = self.OBJECT
            else:
                self.state = self.MOVE
        else:
            msg.angular.z = self.turn_spd
    
    def timer_callback(self):
        msg = Twist()

    
        if self.state == self.MOVE:

            self.get_logger().info("STATE: MOVE")

            if self.is_navigating:
                self.get_logger().info("check1")
                # Check if within 5cm of goal - consider it reached
                if (
                    self.distance_remaining is not None and 
                    self.distance_remaining <= 0.3
                ):
                    self.get_logger().info("Within 5cm of MOVE goal → reached destination")
                    self.cancel_navigation()
                    self.stop(msg)
                    # Remove the reached point
                    self.is_navigating = False
                    self.state = self.SCAN  # Or whatever next state you want
                    return

                # Obstacle detected → switch to AVOID
                if not self.can_move_foward(0.2, 0.3):
                    self.get_logger().warn("Obstacle ahead → switching to AVOID")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.AVOID

                # Object detected while moving → cancel nav & switch
                if len(self.go_to_points) != 0:
                    self.get_logger().warn("Object detected → interrupting MOVE and switching to OBJECT")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.OBJECT

            else:
                self.get_logger().info("check2")
                # No active navigation, so start a new goal
                if len(self.scan_points) == 0:
                    pose = self.get_robot_pose()
                    if pose is None:
                        self.get_logger().warn("Pose not ready yet")
                        return
                    x, y, yaw = pose
                    point_msg = PointStamped()
                    point_msg.header.stamp = self.get_clock().now().to_msg()
                    point_msg.header.frame_id = "map"  # same frame as your TF
                    point_msg.point.x = float(x)
                    point_msg.point.y = float(y)
                    point_msg.point.z = 0.0
                    self.Robot_point.publish(point_msg)
                    self.get_logger().info("check3")
                    return
                self.get_logger().info("check4")
                new_point = self.scan_points[0]
                
                self.send_goala(new_point[0], new_point[1])
                self.is_navigating = True


        # --------------------------------------------------
        # SCAN STATE
        # --------------------------------------------------
        elif self.state == self.SCAN:

            self.get_logger().info("STATE: SCAN")

            if self.turn_counter >= 2 * math.pi:
                self.get_logger().info("Finished one full rotation → switching to MOVE")
                self.stop(msg)
                self.state = self.MOVE
                self.turn_counter = 0
                if self.scan_points:
                    self.scan_points.pop(0) 
                return

            if len(self.go_to_points) > 0:
                self.get_logger().info("Object found during scan → switching to OBJECT")
                self.stop(msg)
                self.turn_counter = 0
                self.state = self.OBJECT
            else:
                msg.angular.z = self.turn_spd
                self.turn_counter += self.turn_spd * self.timer_period


        # --------------------------------------------------
        # OBJECT STATE
        # --------------------------------------------------
        elif self.state == self.OBJECT:

            self.get_logger().info("STATE: OBJECT")

            if not self.go_to_points:
                self.get_logger().warn("OBJECT state but no points left")
                self.state = self.MOVE
                return

            new_point = self.go_to_points[0]

            if self.is_navigating:

                # Obstacle detected
                if not self.can_move_foward(0.2, 0.3):
                    self.get_logger().warn("Obstacle while approaching object → switching to AVOID")
                    self.stop(msg)
                    self.state = self.AVOID

                # Close enough to object?
                if (
                    self.distance_remaining is not None and
                    self.distance_remaining <= self.min_distance_to_object
                ):
                    # Close enough → now face it first!
                    self.get_logger().info("Close to object → switching to TURNTO for alignment")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.TURNTO
                    return


            else:
                self.get_logger().info(f"Sending OBJECT goal to: {new_point}")
                
                self.send_goala(new_point[0], new_point[1])
                self.is_navigating = True


        # --------------------------------------------------
        # AVOID STATE
        # --------------------------------------------------
        elif self.state == self.AVOID:
            self.get_logger().info("STATE: AVOID")
            self.avoid_turn(msg)
        # --------------------------------------------------
        # TURNTO STATE  (face the object before marking it)
        # --------------------------------------------------
        elif self.state == self.TURNTO:

            self.get_logger().info("STATE: TURNTO")

            # Must have an object target
            if not self.go_to_points:
                self.get_logger().warn("TURNTO but no object point available")
                self.stop(msg)
                self.state = self.MOVE
                return

            obj_x, obj_y, obj_z = self.go_to_points[0]

            # Get robot pose (x, y, yaw)
            pose = self.get_robot_pose()
            if pose is None:
                self.get_logger().warn("TURNTO: No pose available yet")
                return

            robot_x, robot_y, robot_yaw = pose

            # --------------------------------------------------
            # 1. Compute the angle robot should face
            # --------------------------------------------------
            desired_yaw = math.atan2(obj_y - robot_y, obj_x - robot_x)

            # Normalize error to [-pi, +pi]
            yaw_error = desired_yaw - robot_yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

            # --------------------------------------------------
            # 2. Check if aligned
            # --------------------------------------------------
            # within ~5 degrees
            if abs(yaw_error) < 0.087:
                self.get_logger().info("TURNTO: Aligned with object → switching to OBJECT")
                self.stop(msg)

                # 2. Publish the object marker (RViz)
                self.publish_marker(obj_x, obj_y, obj_z)
                
                self.get_logger().info("WE SHOULD NOW PUBLISH MARKER")

                # 3. Store object as found
                self.points_found.append(self.go_to_points.pop(0))

                # 4. Switch to MOVE — NO NAV 
                self.state = self.MOVE
                return

            # --------------------------------------------------
            # 3. Rotate toward desired yaw
            # --------------------------------------------------
            k_p = 0.8  # proportional gain (turning sensitivity)
            rot = k_p * yaw_error

            # Limit speed
            rot = max(min(rot, 0.4), -0.4)

            msg.angular.z = rot
            msg.linear.x = 0.0   # no forward movement


        self.publisher_twist.publish(msg)


    def scan_callback(self, msg):
        self.ranges = []
        self.ranges = msg.ranges

    def stop(self,msg):
        msg.linear.x = 0.0
        msg.angular.z = 0.0

def main(args=None):
    rclpy.init(args=args)

    action_client = NavigateToPoseActionClient()


    #rclpy.spin_until_future_complete(action_client, future)

    rclpy.spin(action_client)

    action_client.destroy_node()
    rclpy.shutdown()

    #future = action_client.send_goal(6.02, -1.36)


# Point 1 = 6.02 , -1.36 , 0.00247
# Point 2 = 7.12 , 0.326 , 0.00247
# Point 3 = 7.1  , 4.26  , 0.00247
# Point 4 = 3.14 , 4.27  , 0.00247
# Point 5 = 1.02 , 1.24  , 0.00247