import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from map_msgs.msg import OccupancyGridUpdate
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import cv2


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # QoS for reliable map delivery
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

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
        self.padding_radius = 7
        self.visited_goals = []
        self.visited_frontiers = []

        self.goal_distance_threshold = 3.0
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None

    def map_callback(self, map_msg):
        self.latest_map_msg = map_msg

        # Store resolution and origin values
        self.map_resolution = map_msg.info.resolution
        self.map_origin_x = map_msg.info.origin.position.x
        self.map_origin_y = map_msg.info.origin.position.y

        grid = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.frontiers = self.find_frontiers(grid)
        self.visualize_map(grid)



    def map_update_callback(self, upd):
        if self.latest_map_msg is None:
            return

        w = self.latest_map_msg.info.width
        h = self.latest_map_msg.info.height

        for row in range(upd.height):
            for col in range(upd.width):
                map_index = (upd.y + row) * w + (upd.x + col)
                upd_index = row * upd.width + col
                self.latest_map_msg.data[map_index] = upd.data[upd_index]

        grid = np.array(self.latest_map_msg.data).reshape((h, w))
        self.frontiers = self.find_frontiers(grid)
        self.visualize_map(grid)


    def find_frontiers(self, grid):
        h, w = grid.shape
        frontiers = []

        # Treat -1 AND midpoints (50) as unknown
        unknown_mask = (grid < 0) | (grid == 50)

        for y in range(1, h - 1):
            for x in range(1, w - 1):

                # Only free space
                if grid[y, x] != 0:
                    continue

                neigh = grid[y-1:y+2, x-1:x+2]

                # If any neighbor is unknown → frontier
                if np.any(unknown_mask[y-1:y+2, x-1:x+2]):
                    frontiers.append((x, y))

        return frontiers



    def has_neighbor(self, grid, x, y, value):
        neighbors = grid[y-1:y+2, x-1:x+2].flatten()
        return value in neighbors

    def compute_frontier_scores(self,map_data, frontiers, kernel_size=15):
        scores = []
        radius = kernel_size // 2
        width = map_data.info.width
        height = map_data.info.height
        occ = list(map_data.data)  # occupancy grid (flat list)

        def idx(x, y):
            return y * width + x

        for (x, y) in frontiers:
            score = 0
            # scan kernel around frontier
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if occ[idx(nx, ny)] == -1:   # unknown
                            score += 1
            scores.append(score)

        return scores


    def point_callback(self, point_msg):
        if self.latest_map_msg is None:
            self.get_logger().warn("No map yet")
            return

        robot_x = point_msg.point.x
        robot_y = point_msg.point.y
        mx, my = self.world_to_map(robot_x, robot_y, self.latest_map_msg)

        if not self.frontiers:
            self.get_logger().warn("No valid frontiers!")
            return

        fx, fy = self.select_frontier(mx, my)
        wx, wy = self.map_to_world(fx, fy, self.latest_map_msg)

        goal = PointStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.point.x = float(wx)
        goal.point.y = float(wy)
        self.get_logger().info(f"Selected viable frontier: ({fx}, {fy}) → ({wx:.2f}, {wy:.2f})")

        self.get_logger().info(f"Publishing frontier: ({wx:.2f}, {wy:.2f})")
        self.frontier_pub.publish(goal)


    def select_frontier(self, mx, my):
        if not self.frontiers:
            return None

        map_msg = self.latest_map_msg
        viable = []

        # Build list of viable frontiers
        for fx, fy in self.frontiers:
            # Convert frontier to world coords
            wx, wy = self.map_to_world(fx, fy, map_msg)

            # WORLD coordinates of robot
            robot_wx, robot_wy = self.map_to_world(mx, my, map_msg)

            # WORLD-SPACE FILTERING
            if any(np.hypot(wx - vx, wy - vy) < self.goal_distance_threshold
                for vx, vy in self.visited_goals):
                continue

            # WORLD-SPACE DISTANCE for scoring
            dist = np.hypot(wx - robot_wx, wy - robot_wy)

            viable.append((fx, fy, wx, wy, dist))


        if not viable:
            self.get_logger().warn("No viable frontiers — using fallback")
            fx, fy = self.frontiers.pop(0)
            wx, wy = self.map_to_world(fx, fy, map_msg)
            self.visited_goals.append((wx, wy))
            return fx, fy

        # Extract (fx,fy) list for scoring
        frontiers_xy = [(fx, fy) for (fx, fy, wx, wy, dist) in viable]

        # UNKNOWN SCORE (raw)
        unknown_scores = self.compute_frontier_scores(map_msg, frontiers_xy)

        # DISTANCE SCORE (raw)
        dists = [dist for (_, _, _, _, dist) in viable]

        # Normalize both
        min_u, max_u = min(unknown_scores), max(unknown_scores)
        min_d, max_d = min(dists), max(dists)

        unknown_norm = [(u - min_u) / (max_u - min_u + 1e-6) for u in unknown_scores]
        dist_norm = [1.0 - (d - min_d) / (max_d - min_d + 1e-6) for d in dists]

        # WEIGHTS → tune these:
        W_unknown = 0.7
        W_distance = 0.3

        combined = [
            W_unknown * unknown_norm[i] + W_distance * dist_norm[i]
            for i in range(len(viable))
        ]

        best_idx = int(np.argmax(combined))

        fx, fy, wx, wy, dist = viable[best_idx]
        if (fx, fy) in self.frontiers:
            self.frontiers.remove((fx, fy))

        # Mark visited
        self.visited_frontiers.append((fx, fy))   # map grid coords
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

        for fx, fy in self.frontiers:
            img[fy, fx] = [0, 0, 255]
        for fx, fy in self.visited_frontiers:
            img[fy, fx] = [0, 255, 0]  # green for visited


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
