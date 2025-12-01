import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion, PointStamped
import numpy as np
from map_msgs.msg import OccupancyGridUpdate
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import cv2 

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos   # <-- Use correct QoS here
        )

        # setup action client for navigation
        self.frontiers = []
        self.target_sub = self.create_subscription(
            PointStamped,
            '/robotpoint',
            self.point_callback,
            10
        )
                
        self.map_update_sub = self.create_subscription(
            OccupancyGridUpdate,
            '/map_updates',
            self.map_update_callback,
            10
        )
        self.latest_map_msg = None
        self.Frontier_Publisher = self.create_publisher(PointStamped, '/frontier_point', 10)
    
    def point_callback(self, point_msg):
        if self.latest_map_msg is None:
            self.get_logger().warn("Map not received yet (latest_map_msg is None)")
            return
        self.get_logger().info("check67")
        robot_x = point_msg.point.x
        robot_y = point_msg.point.y
        robot_mx, robot_my = self.world_to_map(robot_x, robot_y, self.latest_map_msg)
        frontier = self.select_frontier(robot_mx, robot_my)

        if frontier is None:
            return
        self.get_logger().info("check68")
        fx,fy = frontier
        fx, fy = self.map_to_world(fx, fy, self.latest_map_msg)
        self.navigate_to(fx, fy)
        
    def map_callback(self, map_msg):

        
        self.latest_map_msg = map_msg
        grid = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.frontiers = self.find_frontiers(grid)
        self.visualize_map(grid) 

    def find_frontiers(self, grid):
        frontiers = []
        for y in range(1, grid.shape[0]-1):
            for x in range(1, grid.shape[1]-1):
                if grid[y,x] == 0 and self.has_unknown_neighbor(grid, x, y):
                    frontiers.append((x,y))
        return frontiers

    def has_unknown_neighbor(self, grid, x, y):
        neighbors = grid[y-1:y+2, x-1:x+2].flatten()
        return -1 in neighbors

    def select_frontier(self, x,y):
        # Choose the closest frontier
        if not self.frontiers:
            return None
        distances = [np.hypot(fx - x, fy - y) for fx, fy in self.frontiers]
        return self.frontiers[np.argmax(distances)]


    def map_update_callback(self, upd):
        if self.latest_map_msg is None:
            return  # haven't received full map yet

        w = self.latest_map_msg.info.width

        # data inside latest_map_msg is flattened
        for row in range(upd.height):
            for col in range(upd.width):
                map_index = (upd.y + row) * w + (upd.x + col)
                upd_index = row * upd.width + col
                self.latest_map_msg.data[map_index] = upd.data[upd_index]
        grid = np.array(self.latest_map_msg.data).reshape(
            (self.latest_map_msg.info.height, self.latest_map_msg.info.width)
        )
        self.frontiers = self.find_frontiers(grid)

    def navigate_to(self, x, y):
        if x is None or y is None:
            return
        goal = PointStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"  # usually the map frame
        goal.point.x = float(x)
        goal.point.y = float(y)
        
        self.get_logger().warn(f"Pose not ready yet x = {x}")
        self.get_logger().warn(f"Pose not ready yet y = {y}")

        self.Frontier_Publisher.publish(goal)
    def map_to_world(self, x_idx, y_idx, map_msg):
        res = map_msg.info.resolution
        origin = map_msg.info.origin
        wx = x_idx * res + origin.position.x + res/2
        wy = y_idx * res + origin.position.y + res/2
        return wx, wy
    
    def world_to_map(self, wx, wy, map_msg):
        res = map_msg.info.resolution
        origin = map_msg.info.origin
        mx = int((wx - origin.position.x) / res)
        my = int((wy - origin.position.y) / res)
        # Clamp to valid indices
        mx = max(0, min(mx, map_msg.info.width - 1))
        my = max(0, min(my, map_msg.info.height - 1))
        return mx, my

        
    def visualize_map(self, grid):
        """
        Create a color visualization of the map with frontiers highlighted
        """
        # Create RGB image
        # -1 = unknown (gray), 0 = free (white), 100 = occupied (black)
        img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        
        # Color the map
        img[grid == -1] = [128, 128, 128]  # Unknown = Gray
        img[grid == 0] = [255, 255, 255]   # Free = White
        img[grid == 100] = [0, 0, 0]       # Occupied = Black
        img[(grid > 0) & (grid < 100)] = [50, 50, 50]  # Uncertain = Dark gray
        
        # Highlight frontiers in red
        for fx, fy in self.frontiers:
            img[fy, fx] = [0, 0, 255]  # Red (BGR format for OpenCV)
        
        # Add a small circle around each frontier for better visibility
        for fx, fy in self.frontiers:
            cv2.circle(img, (fx, fy), 2, (0, 0, 255), -1)
        
        # Flip vertically (OpenCV has origin at top-left, maps have origin at bottom-left)
        img = cv2.flip(img, 0)
        
        # Scale up for better visibility
        scale_factor = 4
        img_large = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_NEAREST)
        
        # Add text overlay
        text = f"Frontiers: {len(self.frontiers)}"
        cv2.putText(img_large, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display image in a window
        cv2.imshow('Frontier Map', img_large)
        cv2.waitKey(1)  # Update window (1ms wait allows processing)

    def map_update_callback(self, upd):
        if self.latest_map_msg is None:
            return  # haven't received full map yet

        w = self.latest_map_msg.info.width

        # data inside latest_map_msg is flattened
        for row in range(upd.height):
            for col in range(upd.width):
                map_index = (upd.y + row) * w + (upd.x + col)
                upd_index = row * upd.width + col
                self.latest_map_msg.data[map_index] = upd.data[upd_index]
        grid = np.array(self.latest_map_msg.data).reshape(
            (self.latest_map_msg.info.height, self.latest_map_msg.info.width)
        )
        self.frontiers = self.find_frontiers(grid)
        
        # Visualize updated map
        self.visualize_map(grid)

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
