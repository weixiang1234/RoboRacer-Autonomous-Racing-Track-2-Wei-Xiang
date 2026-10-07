#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and keep
publishing to the two AutoDRIVE command topics; everything else is yours to
change - rewrite this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then publishes a slow constant throttle with the wheels straight.
It will roll forward off the line and into the first thing in front of it.
That is deliberate and it is the whole point: **there is no algorithm here and
no algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the QoS settings, the workspace
and your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

--------------------------------------------------------------------------
THE CONTROL INTERFACE IS NOT A SPEED REQUEST
--------------------------------------------------------------------------
AutoDRIVE takes a normalised throttle, not a target speed:

    /autodrive/roboracer_1/throttle_command   Float32, [-1, 1]
    /autodrive/roboracer_1/steering_command   Float32, [-1, 1]  ->  +-0.5236 rad

-1 is full reverse, 0 is coasting, +1 is full torque. There is no speed
controller between you and the motor, so "go at 3 m/s" is a control problem you
now own. Whatever algorithm you pick will decide a *target* speed; turning that
into a throttle is a separate job, and it is worth several seconds a lap. See
`docs/04-algorithms.md` §4.1.

--------------------------------------------------------------------------
WHAT YOU MAY READ  (see docs/06-rules.md)
--------------------------------------------------------------------------
    .../lidar            1080 beams over 270 deg, 0.06-10 m
    .../imu              orientation, angular velocity, linear acceleration
    .../left_encoder     wheel angle, 16 PPR x 120
    .../right_encoder
    .../front_camera     RGB image (headless runs do not produce one)
    .../throttle         actuator feedback
    .../steering
    .../ips              ground-truth position - ALLOWED here, see below
    .../odom             ground-truth pose and velocity - ALLOWED, RECOMMENDED
    .../lap_count        .../lap_time  .../last_lap_time  .../best_lap_time
    .../collision_count
    /tf, /tf_static

The AutoDRIVE competition rules mark the pose and race-telemetry topics
"restricted": legal for development, not at race time. **The hackathon does not
adopt that restriction for reading.** Every input above is yours to use during
a scored run, ground-truth pose included. What stays restricted is *publishing*
`/autodrive/reset_command`, which is the referee's (rule 31).
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

# The steering command is normalised; this is what +-1.0 actually means.
MAX_STEERING_RAD = 0.5236


def devkit_qos() -> QoSProfile:
    """Match the QoS the AutoDRIVE bridge publishes and subscribes with.

    RELIABLE / KEEP_LAST(1) / VOLATILE. Get this wrong and the topics simply do
    not connect: a BEST_EFFORT subscriber will never see a RELIABLE publisher's
    messages, and `ros2 topic list` will still cheerfully show the topic.
    """
    return QoSProfile(
        durability=QoSDurabilityPolicy.VOLATILE,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p crawl_throttle:=0.3
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('vehicle_ns', '/autodrive/roboracer_1')
        self.declare_parameter('crawl_throttle', 0.12)    # [-1, 1], open loop
        self.declare_parameter('max_range', 10.0)         # [m] clip the scan here

        self.declare_parameter(
            'centerline_csv',
            '/hackathon/maps/icra26_compete_centerline.csv'
        )
        self.declare_parameter('lookahead_distance', 0.75)
        self.declare_parameter('target_speed', 2.8)

        self.crawl_throttle = self.get_parameter('crawl_throttle').value
        self.max_range = self.get_parameter('max_range').value

        self.lookahead_distance = float(
            self.get_parameter('lookahead_distance').value
        )
        self.target_speed = float(
            self.get_parameter('target_speed').value
        )

        centerline_path = str(
            self.get_parameter('centerline_csv').value
        )

        try:
            route = np.loadtxt(
                centerline_path,
                delimiter=',',
                comments='#',
                dtype=np.float64,
            )

            self.route = route[:, :2]

            self.get_logger().info(
                f'Loaded centerline: {len(self.route)} points '
                f'from {centerline_path}'
            )

        except Exception as exc:
            self.route = None

            self.get_logger().error(
                f'Failed to load centerline: {exc}'
            )

        ns = str(self.get_parameter('vehicle_ns').value).rstrip('/')
        qos = devkit_qos()

        self.throttle_pub = self.create_publisher(Float32, f'{ns}/throttle_command', qos)
        self.steering_pub = self.create_publisher(Float32, f'{ns}/steering_command', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(LaserScan, f'{ns}/lidar', self.scan_callback, qos)
        self.create_subscription(Odometry, f'{ns}/odom', self.odom_callback, qos)
        # Pure Pursuit uses /odom only.
        # self.create_subscription(Point, f'{ns}/ips', self.ips_callback, qos)

        # Latest known pose and speed. Ground truth from the simulator, which
        # the hackathon rules allow you to use - so use it.
        self.position = None      # (x, y) in the `world` frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self._marker_divisor = 0
        self.get_logger().warn(
            'team_driver is up, but this template has no driving logic: it will '
            'crawl straight ahead until it hits something. Implement plan().')

    # ------------------------------------------------------------------
    # Where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def ips_callback(self, msg):
        """Position alone, at the same rate. /odom carries this plus velocity."""
        self.position = (msg.x, msg.y)

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, throttle = self.plan(ranges, angles)
        self.publish(steering, throttle)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering * MAX_STEERING_RAD)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """Track the supplied centreline using Pure Pursuit."""

        if self.position is None or self.route is None:
            return 0.0, 0.0

        x, y = self.position
        n = len(self.route)

        if n < 2:
            return 0.0, 0.0

        # ----------------------------------------------------------
        # 1. Find nearest route point.
        # ----------------------------------------------------------
        dx_all = self.route[:, 0] - x
        dy_all = self.route[:, 1] - y

        dist_sq = (
            dx_all * dx_all
            + dy_all * dy_all
        )

        nearest_index = int(
            np.argmin(dist_sq)
        )

        # ----------------------------------------------------------
        # Dynamic target speed from upcoming route curvature.
        # Look ahead along the route so we slow BEFORE a tight bend.
        # ----------------------------------------------------------
        n = len(self.route)

        max_turn = 0.0

        # Each 4-index segment is roughly 0.8 m on this centerline.
        # Scan approximately the next 5 m of route.
        for offset in range(4, 25, 4):
            i0 = (nearest_index + offset - 4) % n
            i1 = (nearest_index + offset) % n
            i2 = (nearest_index + offset + 4) % n

            v1 = self.route[i1] - self.route[i0]
            v2 = self.route[i2] - self.route[i1]

            h1 = np.arctan2(v1[1], v1[0])
            h2 = np.arctan2(v2[1], v2[0])

            dh = np.arctan2(
                np.sin(h2 - h1),
                np.cos(h2 - h1),
            )

            max_turn = max(max_turn, abs(dh))


        # Straight / gentle section
        target_speed_cmd = self.target_speed

        # Medium corner
        if max_turn > 0.20:
            target_speed_cmd = min(target_speed_cmd, 2.45)

        # Tight corner                  
        if max_turn > 0.40:
            target_speed_cmd = min(target_speed_cmd, 2.45)

        # ----------------------------------------------------------
        # 2. Walk forward along the closed route until lookahead.
        # ----------------------------------------------------------
        accumulated = 0.0
        current_index = nearest_index
        target_index = None

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        for _ in range(n):

            next_index = (
                current_index + 1
            ) % n

            segment_dx = (
                self.route[next_index, 0]
                - self.route[current_index, 0]
            )

            segment_dy = (
                self.route[next_index, 1]
                - self.route[current_index, 1]
            )

            accumulated += math.hypot(
                segment_dx,
                segment_dy,
            )

            current_index = next_index

            if accumulated < self.lookahead_distance:
                continue

            tx = self.route[current_index, 0]
            ty = self.route[current_index, 1]

            dx = tx - x
            dy = ty - y

            local_x = (
                cos_yaw * dx
                + sin_yaw * dy
            )

            if local_x > 0.05:
                target_index = current_index
                break

        if target_index is None:
            return 0.0, 0.0

        # ----------------------------------------------------------
        # 3. Target point in vehicle coordinates.
        # ----------------------------------------------------------
        target_x = self.route[target_index, 0]
        target_y = self.route[target_index, 1]

        dx = target_x - x
        dy = target_y - y

        local_x = (
            cos_yaw * dx
            + sin_yaw * dy
        )

        local_y = (
            -sin_yaw * dx
            + cos_yaw * dy
        )

        lookahead_sq = (
            local_x * local_x
            + local_y * local_y
        )

        if lookahead_sq < 1e-6:
            return 0.0, 0.0

        # ----------------------------------------------------------
        # 4. Pure Pursuit.
        #
        # Track 2 wheelbase = 0.324 m.
        # ----------------------------------------------------------
        curvature = (
            2.0 * local_y
            / lookahead_sq
        )

        steering_angle = math.atan(
            0.324 * curvature
        )

        # Conservative first test.
        steering_angle = float(
            np.clip(
                steering_angle,
                -0.50,
                0.50,
            )
        )

        # AutoDRIVE wants normalised steering.
        steering = (
            steering_angle
            / MAX_STEERING_RAD
        )

        steering = float(
            np.clip(
                steering,
                -1.0,
                1.0,
            )
        )

        # ----------------------------------------------------------
        # 5. Speed controller.
        #
        # AutoDRIVE takes throttle, not target speed.
        # Use feed-forward + proportional correction initially.
        # ----------------------------------------------------------
        speed_error = (
            target_speed_cmd
            - self.speed
        )

        if speed_error < -0.20:
            # Slight overspeed: coast instead of braking.
            throttle = 0.0

        else:
            feedforward = (
                0.05 * target_speed_cmd
            )

            throttle = (
                feedforward
                + 0.12 * speed_error
            )

            throttle = float(
                np.clip(
                    throttle,
                    0.0,
                    0.25,
                )
            )

        # ----------------------------------------------------------
        # Diagnostics at about 4 Hz.
        # ----------------------------------------------------------
        if self._marker_divisor == 0:

            cte = math.sqrt(
                float(
                    dist_sq[nearest_index]
                )
            )

            self.get_logger().info(
                f'PP nearest={nearest_index}, '
                f'target={target_index}, '
                f'cte={cte:.2f}m, '
                f'v={self.speed:.2f}, '
                f'v_target={target_speed_cmd:.2f}, '
                f'steer={steering:+.2f}, '
                f'throttle={throttle:.2f}'
            )

        return steering, throttle
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, throttle):
        self.steering_pub.publish(Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self.throttle_pub.publish(Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'roboracer_1'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
