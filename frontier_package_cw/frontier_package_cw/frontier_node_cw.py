import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from map_msgs.msg import OccupancyGridUpdate
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped

import numpy as np
import cv2

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        # QoS for reliable map delivery
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE, # keep calling util we get enough data (full map)
            durability=DurabilityPolicy.TRANSIENT_LOCAL # late joining subscriptions - ensure late data is added correctly
        )

        # A grid of the map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # Updates the map grid to say what data is inside each cell
        self.map_update_sub = self.create_subscription(
            OccupancyGridUpdate,
            '/map_updates',
            self.map_update_callback,
            10
        )

        self.target_sub = self.create_subscription(
            PointStamped,
            '/robotpoint',
            self.point_callback,
            10
        )

        self.frontier_pub = self.create_publisher(
            PointStamped,
            '/frontier_point',
            10
        )

        self.latest_map_msg = None
        self.frontiers = []
        self.visited_goals = []
        self.visited_frontiers = []
        self.ring_points =  []
        self.goal_distance_threshold = 1.0
 
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None

    def map_callback(self, map_msg):
        self.latest_map_msg = map_msg

        # Store resolution and origin values
        self.map_resolution = map_msg.info.resolution
        self.map_origin_x = map_msg.info.origin.position.x
        self.map_origin_y = map_msg.info.origin.position.y

        grid = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width)) # reshapes map when more points are discovered
        self.frontiers = self.find_frontiers(grid) # includes frontiers
        self.ring_points = self._calculate_safety_ring_points(grid, map_msg)
        self.visualize_map(grid)
        
 

    def _calculate_safety_ring_points(self, grid, map_info):
 
        h, w = grid.shape
        
        obstacle_radius = 6 
        
        initial_obstacle_binary = np.zeros((h, w), dtype=np.uint8)
        initial_obstacle_binary[grid != 0] = 255 
        
        kernel_obs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (obstacle_radius * 2 + 1, obstacle_radius * 2 + 1))
        
        dilated_obstacle_map = cv2.dilate(initial_obstacle_binary, kernel_obs, iterations=1)

        raw_boundary_points = []
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                
                if dilated_obstacle_map[y, x] == 0: 
                    
                    is_adjacent_to_dilated_obstacle = False
                    
                    for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        ny, nx = y + dy, x + dx
                        
                        if dilated_obstacle_map[ny, nx] == 255:
                            is_adjacent_to_dilated_obstacle = True
                            break 
                    
                    if is_adjacent_to_dilated_obstacle:
                        raw_boundary_points.append((x, y))

        boundary_mask = np.zeros((h, w), dtype=np.uint8)
        
        for x, y in raw_boundary_points:
            boundary_mask[y, x] = 255
            
        contours, _ = cv2.findContours(boundary_mask, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE)
        
        largest_contour = None
        max_length = 0

        for contour in contours:
            length = len(contour)
            if length > max_length:
                max_length = length
                largest_contour = contour
                
        ring_points_list = []
        
        if largest_contour is not None:
            points_xy = largest_contour.reshape(-1, 2)
            
            for x, y in points_xy:
                ring_points_list.append((x, y))

        return ring_points_list
    
    def is_goal_reachable(self, goal_msg):
        goal_future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            return False

        if len(result.result.path.poses) == 0:
            return False

        return True

    def map_update_callback(self, upd):
        if self.latest_map_msg is None:
            return

        w = self.latest_map_msg.info.width
        h = self.latest_map_msg.info.height

        for row in range(upd.height):
            for col in range(upd.width):
                map_index = (upd.y + row) * w + (upd.x + col) # what is this doing? - is this just putting the 0, 1, -1 into the correct place when the map grows in size?
                upd_index = row * upd.width + col
                self.latest_map_msg.data[map_index] = upd.data[upd_index]

        grid = np.array(self.latest_map_msg.data).reshape((h, w)) # If so, what is the point in this?
        self.frontiers = self.find_frontiers(grid)
        self.ring_points = self._calculate_safety_ring_points(grid, upd)
        self.visualize_map(grid)

    def find_frontiers(self, grid):
        h, w = grid.shape
        frontiers = []

        # Treat -1 as unknown
        unknown_mask = grid < 0

        for y in range(1, h - 1):
            for x in range(1, w - 1):

                # Only free space
                if grid[y, x] != 0:
                    continue
                
                # If any neighbor is unknown = frontier
                if np.any(unknown_mask[y-2:y+2, x-2:x+2]):
                    if all((fx - x)**2 + (fy - y)**2 > 9 for fx, fy in frontiers): # Prevents frontiers from being duplicated, ensures a single line
                        frontiers.append((x, y))

        return frontiers

    # Nuked has has_neigbor because it's never used 
    
    def idx(self, x, y, width):
        return y * width + x

    def compute_frontier_scores(self,map_data, frontiers, kernel_size=15):
        scores = []
        radius = kernel_size // 2
        width = map_data.info.width
        height = map_data.info.height
        occ = list(map_data.data)  # occupancy grid is the current grid

        for (x, y) in frontiers:
            score = 0
            # scan kernel around frontier
            for dy in range(-radius, radius+1): 
                for dx in range(-radius, radius+1): # for the circle around our current point we get
                    nx = x + dx # x of the point
                    ny = y + dy # y of the point
                    if 0 <= nx < width and 0 <= ny < height: # Make sure it's within the maps boundaries
                        if occ[self.idx(nx, ny, width)] == -1:   # Is the point on the grid -1
                            score += 1 # If it is the score is increased, increasing it's priority
            scores.append(score)

        return scores

    def point_callback(self, point_msg):
        if self.latest_map_msg is None:
            self.get_logger().warn("No map yet")
            return

        robot_x = point_msg.point.x
        robot_y = point_msg.point.y
        
        mx, my = self.world_to_map(robot_x, robot_y, self.latest_map_msg) # Robots x and y based in grid co-ordinates

        if not self.frontiers:
            self.get_logger().warn("No valid frontiers!") # All frontiers are currently found
            return

        fx, fy = self.select_frontier(mx, my) # Get the frontiers co-ordinates based on the robots current position, best frontier to go to based on said values
        wx, wy = self.map_to_world(fx, fy, self.latest_map_msg) # Converting fx and fy from map to world based co-ordinates

        goal = PointStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.point.x = float(wx) # Attaching x
        goal.point.y = float(wy) # Attaching y
        self.get_logger().info(f"Selected viable frontier: ({fx}, {fy}) = ({wx:.2f}, {wy:.2f})")

        self.get_logger().info(f"Publishing frontier: ({wx:.2f}, {wy:.2f})")
        self.frontier_pub.publish(goal) # Publish frontier position

    def select_frontier(self, mx, my):
        if not self.frontiers: # If there's no frontiers return
            return None

        map_msg = self.latest_map_msg
        viable = []

        for fx, fy in self.frontiers: # For every co-ordinate in frontiers
            
            wx, wy = self.map_to_world(fx, fy, map_msg) # Get there real world values
            robot_wx, robot_wy = self.map_to_world(mx, my, map_msg) # Along with the robots
            
            if any(np.hypot(wx - vx, wy - vy) < self.goal_distance_threshold for vx, vy in self.visited_goals): # For evert goal it's visited get it's hypo from itself to the goal. 
                continue
            
            dist = np.hypot(wx - robot_wx, wy - robot_wy) # Calculate it's distance from the point to the robot

            viable.append((fx, fy, wx, wy, dist)) # Create a list which has all the valid points in it 

        if not viable:
            self.get_logger().warn("No viable frontiers — using fallback")
            fx, fy = self.frontiers.pop(0) # Change this code
            wx, wy = self.map_to_world(fx, fy, map_msg)
            self.visited_goals.append((wx, wy))
            return fx, fy

        frontiers_xy = [(fx, fy) for (fx, fy, wx, wy, dist) in viable]
        unknown_scores = self.compute_frontier_scores(map_msg, frontiers_xy)
        dists = [dist for (_, _, _, _, dist) in viable]
        min_u, max_u = min(unknown_scores), max(unknown_scores)
        min_d, max_d = min(dists), max(dists)

        unknown_norm = [(u - min_u) / (max_u - min_u + 1e-6) for u in unknown_scores] # 1e-6 needed to not divide by 0
        dist_norm = [1.0 - (d - min_d) / (max_d - min_d + 1e-6) for d in dists]
        
        W_unknown = 0.6 # weighted towards unknown
        W_distance = 0.4 # more than distances

        combined = [
            W_unknown * unknown_norm[i] - W_distance * dist_norm[i]
            for i in range(len(viable))
        ]

        best_idx = int(np.argmax(combined))

        fx, fy, wx, wy, dist = viable[best_idx]
        min_ring_dist = float('inf')
        closest_ring_world_coords = None

       
        for rx, ry in self.ring_points:
           
            rwx, rwy = self.map_to_world(rx, ry, map_msg)

           
            ring_dist = np.hypot(wx - rwx, wy - rwy)
            

            if ring_dist < min_ring_dist:
                min_ring_dist = ring_dist
                closest_ring_world_coords = (rwx, rwy)

        if closest_ring_world_coords is None:
            self.get_logger().warn("Safety ring is empty or calculation failed. Returning original frontier.")
            closest_ring_world_coords = (wx, wy)

        # Update (fx, fy) and (wx, wy) to be the closest point on the safety ring
        wx, wy = closest_ring_world_coords
        fx, fy = self.world_to_map(wx, wy, map_msg)

        
        if (fx, fy) in self.frontiers:
            self.frontiers.remove((fx, fy))
            
        self.visited_frontiers.append((fx, fy))  
        self.visited_goals.append((wx, wy)) 
        return fx, fy

    def world_to_map(self, wx, wy, map_msg):
        res = map_msg.info.resolution
        origin = map_msg.info.origin
        mx = int((wx - origin.position.x) / res)
        my = int((wy - origin.position.y) / res)
        mx = max(0, min(mx, map_msg.info.width - 1))
        my = max(0, min(my, map_msg.info.height - 1))
        return mx, my

    def map_to_world(self, x_idx, y_idx, map_msg):
        res = map_msg.info.resolution
        origin = map_msg.info.origin
        wx = x_idx * res + origin.position.x + res / 2.0
        wy = y_idx * res + origin.position.y + res / 2.0
        return wx, wy
    

    def visualize_map(self, grid):

        img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        
        img[grid == -1] = [128, 128, 128]   
        img[grid == 0] = [255, 255, 255]   
        img[grid == 100] = [0, 0, 0]       
        img[(grid > 0) & (grid < 100)] = [50, 50, 50] 
       
        if self.latest_map_msg:
            for mx, my in self.ring_points: 
            

                if 0 <= my < img.shape[0] and 0 <= mx < img.shape[1]:

                    img[my, mx] = [0, 255, 255]

        
        for fx, fy in self.frontiers:
            img[fy, fx] = [255, 0, 0]  
        for fx, fy in self.visited_frontiers:
            img[fy, fx] = [0, 255, 0] 


     
        img = cv2.flip(img, 0)
        img_large = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        cv2.putText(img_large, f"Frontiers: {len(self.frontiers)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frontier Map", img_large)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
