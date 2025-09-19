import json
import numpy as np

def load_instances(json_file):
    """Load objects from instances.json."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["annotations"]

def invert_pose(c2w):
    """Invert camera-to-world to world-to-camera."""
    return np.linalg.inv(c2w)

def project_point(world_point, W2C, intrinsics, width, height):
    """Project a 3D world point to 2D pixel coords."""
    Pw = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    Pc = W2C @ Pw
    Xc, Yc, Zc = Pc[:3]

    if Zc <= 0:
        return None

    fx, fy, cx, cy = intrinsics
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    if 0 <= u < width and 0 <= v < height:
        return (u, v)
    else:
        return None

def get_bbox_corners(bbox):
    """Return 8 corners of a 3D bounding box."""
    min_x, min_y, min_z = bbox["min_x"], bbox["min_y"], bbox["min_z"]
    max_x, max_y, max_z = bbox["max_x"], bbox["max_y"], bbox["max_z"]
    return [
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ]

def visible_objects_from_pose(instances_file, pose_matrix, intrinsics, width, height):
    """Return visible objects from a given pose."""
    instances = load_instances(instances_file)
    W2C = invert_pose(pose_matrix)

    visible_objects = []

    for obj in instances:
        bbox = obj["bounding_box"]
        corners = get_bbox_corners(bbox)

        projected = []
        for corner in corners:
            uv = project_point(corner, W2C, intrinsics, width, height)
            if uv is not None:
                projected.append(uv)

        if len(projected) >= 2:
            us, vs = zip(*projected)
            x1, y1, x2, y2 = min(us), min(vs), max(us), max(vs)
            visible_objects.append({
                "object": obj["class_name"],
                "instance_id": obj["instance_id"],
                "bbox_2d": [x1, y1, x2, y2]
            })

    return visible_objects

# --------------------------
# Main test
# --------------------------
if __name__ == "__main__":
    instances_file = "/Users/abu/Downloads/Master-Project-Dataset-Creation-main 2/data/aggregations/scene0000_00_instances.json"

    pose_matrix = np.array([
         [-0.199371, -0.433294,  0.878924, 2.499828],
         [-0.978237,  0.140613, -0.152579, 3.603600],
         [-0.057476, -0.890216, -0.451898, 1.420342],
         [ 0.000000,  0.000000,  0.000000, 1.000000]
    ])

    # Example ScanNet intrinsics
    intrinsics = (577.870605, 577.870605, 319.5, 239.5)
    width, height = 640, 480

    visible = visible_objects_from_pose(instances_file, pose_matrix, intrinsics, width, height)

    print("Visible objects in this pose:")
    if not visible:
        print("⚠️ No objects were visible")
    for obj in visible:
        print(obj)
