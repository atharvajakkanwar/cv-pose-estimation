import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

def get_hier_contours(image):
    """
    Returns the contours and their heirarachy for a given image.
    """
    # Constants
    thresh = 200
    thresh_max = 255
    canny_min = 100
    canny_max = 200

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, thresh_img = cv2.threshold(gray_img, thresh, thresh_max, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh_img, canny_min, canny_max)
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.squeeze()
    return hierarchy, contours

class Contour:
    """
    Represents a contour with its number, the contour data and its centroid.
    """
    def __init__(self, num, contour, centroid):
        self.num = num
        self.contour = contour
        self.centroid = centroid

def get_centroid(contour):
    """
    Returns the centroid for a given contour.
    """
    m = cv2.moments(contour)
    if m['m00'] != 0:
        centroid_x =int(m['m10']/m['m00'])
        centroid_y =int(m['m01']/m['m00'])
    return np.array((centroid_x, centroid_y))

def get_position_markers(hierarchy, contours):
    """
    Returns an array of Contour classes for the position markers of the QR code. 
    """
    #Constants
    nesting_max = 5
    conts = []
    for h in range(len(hierarchy)):
        i = h 
        nesting = 0
        while hierarchy[i][2] != -1:
            i = hierarchy[i][2]
            nesting += 1
        if nesting == nesting_max:
            conts.append(Contour(h, contours[h], get_centroid(contours[h])))
#     corner_marker_contours = np.asarray((conts[0].contour,conts[1].contour,conts[2].contour))
    return np.asarray(conts)

def get_dist(point_a, point_b):
    """
    Returns the distance between two 2D coordinates.
    """
    a = point_a.astype(np.float32)
    b = point_b.astype(np.float32)
    return np.linalg.norm(a - b)

def vert_opp_largest_side(C1, C2, C3):
    """
    Given 3 points(Contours in this case), returns the 3 contours rearranged
    in such a way that the cetroid of first returned contour is opposite to the largest side 
    in the triangle formed by the three centroids. 
    """
    P1 = C1.centroid
    P2 = C2.centroid
    P3 = C3.centroid
    
    P1P2 = get_dist(P1, P2)
    P2P3 = get_dist(P2, P3)
    P3P1 = get_dist(P3, P1)
    
    if P1P2 > P2P3 and P1P2 > P3P1:
        return [C3, C1, C2]
    elif P2P3 > P1P2 and P2P3 > P3P1:
        return [C1, C2, C3]
    elif P3P1 > P1P2 and P3P1 > P2P3:
        return [C2, C1, C3]

def get_midpoint(point_a, point_b):
    """
    Returns the midpoint of the line segment joining the given 2 points.
    """
    return np.asarray(((point_a[0]+point_b[0])/2, (point_a[1]+point_b[1])/2))

def get_max_dist(cnt, point):
    """
    Returns the pointfrom the contour that is farthest from the given point.
    """
    l = np.asarray(cnt[cnt[:,:,0].argmin()][0])
    r = np.asarray(cnt[cnt[:,:,0].argmax()][0])
    t = np.asarray(cnt[cnt[:,:,1].argmin()][0])
    b = np.asarray(cnt[cnt[:,:,1].argmax()][0])
    
    req = [l,r,t,b]
    
    max_dist = -1;
    max_pts = point
    for p in req:
        dist = get_dist(p, point)
        if max_dist < dist:
            max_dist = dist
            max_pts = p
    return max_pts.squeeze()

def get_world_points(actual_dim):
    """
    Returns the world coordinates of the given the actual dimension of the image, 
    assuming that the world center is the object center.
    """
    world_points = np.array([(-actual_dim / 2, actual_dim / 2, 0.0),
                                 (actual_dim / 2, actual_dim / 2, 0.0),
                                 (-actual_dim / 2, -actual_dim / 2, 0.0),
                                 (0,0, 0.0)
                                 ])
    return world_points

def get_intrinsic(image):
    """
    Returns the intrinsic matrix of the camera based on the above assumptions.
    """
    focal_length = image.shape[0] #px=cm
    cam_center_x = image.shape[0] / 2
    cam_center_y = image.shape[1] / 2

    intrinsic = np.array([[focal_length, 0, cam_center_y],
                         [0, focal_length, cam_center_x],
                         [0, 0, 1]
                         ], dtype="double")
    return intrinsic

def get_cam_vectors(image, image_points, actual_dim, distortion):
    """
    Returns the rotaional and translational vectro of a camera, 
    given the image, known image point and actual dimensions of the object.
    """
    world_points = get_world_points(actual_dim = 8.8) #cm=px
    intrinsic = get_intrinsic(image)
    flag, rotation_vector, translation_vector = cv2.solvePnP(world_points,
                                                         image_points,
                                                         intrinsic,
                                                         distortion,
                                                         flags=cv2.SOLVEPNP_ITERATIVE)
    return rotation_vector, translation_vector

def get_euler_angles(R):
    import math
    """
    Converts the rotational matrix to euler angles in degrees.
    (yaw, pitch , roll)
    """
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0]))
    z = math.atan2(R[1,0], R[0,0])
    return np.degrees(x), np.degrees(y), np.degrees(z)

def get_cam_data(img_path):
    image = cv2.imread(img_path)
    hierarchy, contours = get_hier_contours(image)
    conts = get_position_markers(hierarchy, contours)
    A, B, C = vert_opp_largest_side(conts[0], conts[1], conts[2])
    mid = get_midpoint(B.centroid, C.centroid)
    x = get_max_dist(A.contour, mid)
    y = get_max_dist(B.contour, mid)
    z = get_max_dist(C.contour, mid)
    image_points = np.array([x, z, y, mid])

    distortion = np.zeros((4, 1))
    actual_dim = 8.8
    rotation_vector, translation_vector = get_cam_vectors(image, image_points, actual_dim, distortion)
    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

    pose = np.matmul(-rotation_matrix.transpose(), translation_vector)
    pose = pose.squeeze()
    orientation = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)
    angles = get_euler_angles(rotation_matrix)
    return pose, orientation, angles

def plot_cam(ax,data, img):
    """
    Plotting the camera in real world, given its pose, orientation and angles.
    """
    pose = np.around(data[0], decimals=4)
    orientation = np.around(data[1], decimals=4)
    angles = np.around(data[2], decimals=4)

    ax.scatter([pose[0]], [pose[1]], [pose[2]])
    ax.text(pose[0]-15, pose[1], pose[2], img)
    label_pose = ("Pose", '%.3f' % pose[0], '%.3f' % pose[1], '%.3f' % pose[2])
    ax.text(pose[0], pose[1], pose[2], label_pose)
    label_orient = ("Orientation Vector", '%.3f' % orientation[0], '%.3f' % orientation[1], '%.3f' % orientation[2])
    ax.text(pose[0], pose[1], pose[2]+5, label_orient)
    label_angle = ("Yaw,Pitch,Roll", '%.3f' % angles[0], '%.3f' % angles[1], '%.3f' % angles[2])
    ax.text(pose[0], pose[1], pose[2]+10, label_angle)
    ax.plot([pose[0], 0],[pose[1], 0], zs=[pose[2], 0])


import sys
import os

def main(argv):
    if len(argv) < 3:
        raise Exception("Insufficient input arguments. \n "
                        "USE: python pose.py [input_images_dir_path] [original_image_path] [original_imgage_dimension]")
    directory = os.fsencode(str(argv[1]))
    image_paths = []
    image_names = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_paths.append(os.path.join(directory.decode("utf-8"), filename))
            image_names.append(filename)
            continue
        else:
            continue

    actual_dim = float(argv[3])
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # axis_length = max(100, np.amax(pose))
    axis_length = 100
    ax.set_xlim3d(-axis_length, axis_length)
    ax.set_ylim3d(-axis_length, axis_length)
    ax.set_zlim3d(0, axis_length)

    # Plot camera
    for img, name in zip(image_paths, image_names):
        plot_cam(ax, get_cam_data(img), name)

    # Plot QR Code
    pattern = cv2.imread(argv[2])
    pattern = cv2.flip(cv2.transpose(pattern), 0)

    qr_x, qr_y = np.meshgrid(np.linspace(-actual_dim, actual_dim, pattern.shape[0]),
                         np.linspace(-actual_dim, actual_dim, pattern.shape[0]))
    X = qr_x
    Y = qr_y
    Z = 0
    ax.plot_surface(X, Y, Z, facecolors=pattern / 255., shade=False)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == '__main__':
    main(sys.argv)




