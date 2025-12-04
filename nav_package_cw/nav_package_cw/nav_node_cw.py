import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
from action_msgs.msg import GoalStatus

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
        self.current_pose = None 
        self.stuck_distance_threshold = 0.2 
        self.timer_stuck = self.create_timer(self.timer_period_stuck, self.timer_callback_stuck)
        
        self.ranges = []
        self.scan_points = [] # could be, maybe, idk chat (2.4, 0.0), (1.41, -3.41), (-0.47, -0.57), (-1.59, -3.03), (-1.58, 2.98), (1.01, 3.99),
                            #(3.68, 2.01), (5.51, 3.01), (5.41, -2.07)
        self.go_to_points = []
        self.points_found = []# list of points found 
        self.is_navigating = False
        self.current_goal_handle = None
        self.marker_counter = 0
        
        self.turn_spd = math.pi / 5.0
        self.turn_counter = 0.0
        self.distance_remaining = None
        self.min_distance_to_object = 0.4
        self.min_wall_distance = 0.2
        
        self.marker_pub = self.create_publisher(Marker, "/eye_target_marker", 10)
        self.Robot_point = self.create_publisher(PointStamped, '/robotpoint', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
    def frontier_callback(self, msg):
        x = msg.point.x
        y = msg.point.y
    
        self.scan_points.append((x, y))

    def timer_callback_stuck(self):
        if self.state == self.SCAN or self.state == self.TURNTO: 
            return
        
        x, y, yaw = self.get_robot_pose()

        if x is None:
            return

        if self.prev_pos is None:
            self.prev_pos = (x, y)
            return

        px, py = self.prev_pos

        # Euclidean distance moved since last check
        dist = np.hypot(x - px, y - py)

        # Debug print
        self.get_logger().info(f"[STUCK CHECK] Moved: {dist:.3f} m")

        # If robot moved less than threshold = it's stuck
        if dist < self.stuck_distance_threshold:
            self.get_logger().warn("Robot appears STUCK = Canceling current goal!")

            try:
                if self.state == self.MOVE:
                    # cancel the goal (assuming you have nav2 action client named self.nav_client)
                    self.cancel_navigation()
                    if self.scan_points:
                        self.scan_points.pop(0)
                elif self.state == self.OBJECT:
                    self.cancel_navigation()
                    self.state = self.TURNTO
            except Exception as e:
                self.get_logger().error(f"Failed to cancel goal: {e}")

        # Update for next cycle
        self.prev_pos = (x, y)

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time()) # This is getting the robot's pos and map to work out final pos
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation # This is giving the robots quart values
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except Exception as e:
                self.get_logger().error(f"Failed to calculate robots pose: {e}")
                return None
    
    def send_goala(self, x, y, z = 0.0): # This need's to be changed bro we can't just be slapping an A on it
        
        self._action_client.wait_for_server()

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.w = 1.0  # Neutral orientation
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0 # Should we just remove this? 
        
        goal_msg = NavigateToPose.Goal()
        
        goal_msg.pose = pose
        goal_msg.behavior_tree = ''
        
        self.get_logger().info('point %f %f' % (x, y))
        self.publish_marker(x,y,z)
        
        send_goal_future = self._action_client.send_goal_async( # How is this called multiple times?
            goal_msg,
            feedback_callback=self.feedback_callback # This is called multiple times to get the self.distance_remaining value
        )
        
        send_goal_future.add_done_callback(self.goal_response_callback) # This is only ever called once - accepts or rejects the goal
        return send_goal_future

    def cancel_navigation(self):
        self.is_navigating = False
        if self.current_goal_handle is None:
            self.get_logger().info("No goal to cancel.")
            return

        self.get_logger().warn("Cancelling current goal.")
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
        self._get_result_future.add_done_callback(self.get_result_callback) # Pass results from next function to the future goal 

    def get_result_callback(self, future):
        result = future.result().result  # Proper extraction of result object
        if len(result.result.path.poses) == 0:
            self.get_logger().log("invalid move")
        if result.result == 0: # Worked correctly
            if self.state == self.MOVE:
                self.state = self.SCAN
                self.get_logger().log("Move Finished, Scanning")
                
            if self.state == self.OBJECT:
                self.state = self.MOVE
                self.get_logger().l("object found, publishing")
                new_point = self.go_to_points[0]
                self.publish_marker(new_point[0], new_point[1], new_point[2])
                self.points_found.append(self.go_to_points.pop(0))
                
        elif result.result == 1: # Navigation has been cancelled
            self.get_logger().error("Navigation cancelled") 
            
        # Why would it fail?
            
        elif result.result == 2: # Navigation has failed
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
        result_msg = future.result().result      # the NavigateToPose::Result message
        status = future.result().status          # goal completion status

        # SUCCESS
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation succeeded")

            if self.state == self.MOVE:
                self.state = self.SCAN
                self.get_logger().info("Move Finished, Scanning")

            elif self.state == self.OBJECT:
                self.state = self.MOVE
                self.get_logger().info("object found, publishing")
                new_point = self.go_to_points[0]
                self.publish_marker(new_point[0], new_point[1], new_point[2])
                self.points_found.append(self.go_to_points.pop(0))
        
        # CANCELED
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error("Navigation CANCELLED")

        # FAILED / ABORTED
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("Navigation FAILED — goal unreachable or planning failed!")

    def target_received_callback(self, pose_msg): # New point to move to 
       
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z

        self.go_to_points.append((x, y, z))## adds target to list

    def feedback_callback(self, feedback_msg):
        #feedback = feedback_msg.feedback
        self.distance_remaining = feedback_msg.feedback.distance_remaining # Distance left to travel to goal/point
        self.get_logger().info("Received feedback: " + str(self.distance_remaining))

        if feedback_msg.feedback.distance_remaining == float('inf'):
            self.get_logger().warn("Planner cannot find a valid path anymore!#############################")

    def publish_marker(self, x, y, z): # This just publishs the marker to the rviz map, like a bin or a fire hyrdrant etc
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
        elif z == 1:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        self.get_logger().info("Publishing")
        self.marker_pub.publish(marker)

    def avoid_turn(self,msg):
        if (self.can_move_foward(0.2, 0.3)): # Robots minimum cone of vision
   
            self.stop(msg)
            if len(self.go_to_points) != 0:
                self.state = self.OBJECT
            else: # May need to be swapped around?
                self.state = self.MOVE
        else:
            msg.angular.z = self.turn_spd
    
    def timer_callback(self):
        msg = Twist()

        if self.state == self.MOVE:
            self.get_logger().info("STATE: MOVE")

            if self.is_navigating:
                # Check if within 5cm of goal - consider it reached
                if (self.distance_remaining is not None and self.distance_remaining <= 0.3):
                    self.get_logger().info("Within 5cm of MOVE goal = reached destination")
                    self.cancel_navigation()
                    self.stop(msg)
                    # Remove the reached point
                    self.state = self.SCAN  # Or whatever next state you want
                    return

                # Obstacle detected = switch to AVOID
                if not self.can_move_foward(0.2, 0.3): # um chat, why these values again?
                    self.get_logger().warn("Obstacle ahead = switching to AVOID")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.AVOID

                # Object detected while moving = cancel nav & switch
                if len(self.go_to_points) != 0:
                    self.get_logger().warn("Object detected = interrupting MOVE and switching to OBJECT")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.OBJECT

            else:
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
                    self.Robot_point.publish(point_msg) # The point the robots going to
                    return
                
                new_point = self.scan_points[0]
                self.send_goala(new_point[0], new_point[1])
                self.is_navigating = True

        elif self.state == self.SCAN:
            self.get_logger().info("STATE: SCAN")

            if self.turn_counter >= 2 * math.pi: # 2pi is equal to one full rotation
                self.get_logger().info("Finished one full rotation = switching to MOVE")
                self.stop(msg)
                self.state = self.MOVE
                self.turn_counter = 0
                if self.scan_points:
                    self.scan_points.pop(0) 
                return

            if len(self.go_to_points) > 0:
                self.get_logger().info("Object found during scan = switching to OBJECT")
                self.stop(msg)
                self.turn_counter = 0
                self.state = self.OBJECT
            else:
                msg.angular.z = self.turn_spd
                self.turn_counter += self.turn_spd * self.timer_period

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
                    self.get_logger().warn("Obstacle while approaching object = switching to AVOID")
                    self.stop(msg)
                    self.state = self.AVOID

                # Close enough to object? - cant go forward -> stop movement -> set state to turn to face the object
                if (self.distance_remaining is not None and self.distance_remaining <= self.min_distance_to_object):
                    # Close enough = now face it first! You got it chat!
                    self.get_logger().info("Close to object = switching to TURNTO for alignment")
                    self.cancel_navigation()
                    self.stop(msg)
                    self.state = self.TURNTO
                    return
            else:
                self.get_logger().info(f"Sending OBJECT goal to: {new_point}")
                self.send_goala(new_point[0], new_point[1])
                self.is_navigating = True

        elif self.state == self.AVOID:
            self.get_logger().info("STATE: AVOID")
            self.avoid_turn(msg)

        elif self.state == self.TURNTO:
            self.get_logger().info("STATE: TURNTO")
            # Must have an object target
            if not self.go_to_points:
                self.get_logger().warn("TURNTO but no object point available")
                self.stop(msg)
                self.state = self.MOVE
                return

            obj_x, obj_y, obj_z = self.go_to_points[0] # will be the object x,y,z cords

            # Get robot pose (x, y, yaw)
            pose = self.get_robot_pose()
            if pose is None:
                self.get_logger().warn("TURNTO: No pose available yet")
                return

            robot_x, robot_y, robot_yaw = pose

            desired_yaw = math.atan2(obj_y - robot_y, obj_x - robot_x)
            # Normalize error to [-pi, +pi]
            yaw_error = desired_yaw - robot_yaw # Get angle from robot to desired yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi # Based on that what do I need to do to get there

            # within ~5 degrees
            if abs(yaw_error) < 0.087: # This is 5 degrees in radians?
                self.get_logger().info("TURNTO: Aligned with object = switching to OBJECT")
                self.stop(msg)

                self.publish_marker(obj_x, obj_y, obj_z)
                self.get_logger().info("WE SHOULD NOW PUBLISH MARKER")
                self.points_found.append(self.go_to_points.pop(0))
                self.state = self.MOVE
                return

            k_p = 0.8 # proportional gain (turning sensitivity)
            rot = k_p * yaw_error # The bigger the yaw_error the faster you move
            rot = max(min(rot, 0.4), -0.4) # But there is a cap 

            msg.angular.z = rot
            msg.linear.x = 0.0 # no forward movement

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
    rclpy.spin(action_client)
    action_client.destroy_node()
    rclpy.shutdown()
    