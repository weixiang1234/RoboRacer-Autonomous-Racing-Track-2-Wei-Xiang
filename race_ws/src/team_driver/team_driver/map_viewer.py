#!/usr/bin/env python3

import numpy as np
import rclpy

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from visualization_msgs.msg import Marker


MAP_PATH = '/hackathon/maps/icra26_compete.pgm'
CENTERLINE_PATH = '/hackathon/maps/icra26_compete_centerline.csv'

RESOLUTION = 0.050
ORIGIN_X = -7.960
ORIGIN_Y = -13.230

OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196


def read_pgm(path):
    """Read the official PGM without requiring OpenCV/Pillow."""

    with open(path, 'rb') as f:
        magic = f.readline().strip()

        def next_line():
            while True:
                line = f.readline()

                if not line:
                    raise ValueError('Unexpected end of PGM header')

                line = line.strip()

                if line and not line.startswith(b'#'):
                    return line

        width, height = map(int, next_line().split())
        max_value = int(next_line())

        if max_value > 255:
            raise ValueError('Only 8-bit PGM files are supported')

        if magic == b'P5':
            data = np.frombuffer(
                f.read(width * height),
                dtype=np.uint8,
            )

        elif magic == b'P2':
            data = np.asarray(
                [int(x) for x in f.read().split()],
                dtype=np.uint8,
            )

        else:
            raise ValueError(f'Unsupported PGM format: {magic}')

    if data.size != width * height:
        raise ValueError(
            f'Expected {width * height} pixels, got {data.size}'
        )

    return data.reshape((height, width))


class MapViewer(Node):

    def __init__(self):
        super().__init__('map_viewer')

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            latched_qos,
        )

        self.line_pub = self.create_publisher(
            Marker,
            '/driver/centerline',
            latched_qos,
        )

        self.map_msg = self.make_map()
        self.line_msg = self.make_centerline()

        # Publish once after ROS discovery has had a moment to start.
        self.timer = self.create_timer(0.5, self.publish_once)

    def make_map(self):
        image = read_pgm(MAP_PATH)

        # ROS map_server convention:
        # probability = (255 - pixel) / 255
        probability = (
            255.0 - image.astype(np.float32)
        ) / 255.0

        occupancy = np.full(
            image.shape,
            -1,
            dtype=np.int8,
        )

        occupancy[
            probability > OCCUPIED_THRESH
        ] = 100

        occupancy[
            probability < FREE_THRESH
        ] = 0

        # PGM image starts at top-left.
        # OccupancyGrid starts at bottom-left.
        occupancy = np.flipud(occupancy)

        msg = OccupancyGrid()

        msg.header.frame_id = 'world'

        msg.info.resolution = RESOLUTION
        msg.info.width = image.shape[1]
        msg.info.height = image.shape[0]

        msg.info.origin.position.x = ORIGIN_X
        msg.info.origin.position.y = ORIGIN_Y
        msg.info.origin.orientation.w = 1.0

        msg.data = occupancy.flatten().tolist()

        return msg

    def make_centerline(self):
        route = np.loadtxt(
            CENTERLINE_PATH,
            delimiter=',',
            comments='#',
        )

        marker = Marker()

        marker.header.frame_id = 'world'
        marker.ns = 'official_centerline'
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.06

        marker.color.g = 1.0
        marker.color.a = 1.0

        for x, y in route[:, :2]:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.03

            marker.points.append(point)

        # Close the loop visually.
        if marker.points:
            marker.points.append(marker.points[0])

        return marker

    def publish_once(self):
        now = self.get_clock().now().to_msg()

        self.map_msg.header.stamp = now
        self.line_msg.header.stamp = now

        self.map_pub.publish(self.map_msg)
        self.line_pub.publish(self.line_msg)

        self.get_logger().info(
            'Published /map and /driver/centerline'
        )

        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    node = MapViewer()

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