import numpy as np
import cv2
from skimage import io
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
def is_line_crossing_other_cc(pt1, pt2, labeled_img):
    """
    Checks if a line between two points crosses a different connected component (cc) than the ones the points belong to.
    """
    line_points = bresenham_line(pt1[1], pt1[0], pt2[1], pt2[0])  # x, y swapped for bresenham_line
    cc1 = labeled_img[pt1[0], pt1[1]]
    cc2 = labeled_img[pt2[0], pt2[1]]

    for x, y in line_points:
        if labeled_img[y, x] not in [0, cc1, cc2]:  # 0 is for background, cc1 & cc2 are the components of pt1 & pt2
            return True  # Line crosses a different cc
    return False

# 加载图像并应用骨架化
def load_and_skeletonize_image(file_path):
    image = io.imread(file_path, as_gray=True)
    image = (image > 0).astype(int)  # 确保图像是二值化的
    skeleton = skeletonize(image)
    return skeleton


# 提取端点和交叉点
def extract_points(skel):
    cross_kernel = np.array([[1, 1, 1],
                             [1, 10, 1],
                             [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, cross_kernel)
    endpoints = filtered == 11  # 端点条件
    crosspoints = filtered >= 13  # 交叉点条件
    return endpoints, crosspoints


# 将点的布尔数组转换为坐标列表
def points_to_coords(points):
    return np.column_stack(np.where(points))


def find_mutual_closest_pairs_v2(pts1, pts2, labeled_img):
    # 构建KD树
    tree1 = cKDTree(pts1)

    # 对pts1中每个点找到pts1中的最近点，因为是同一组点，需要找到第二近的点
    dists, idx = tree1.query(pts1, k=2)
    # 第二近的点的索引
    nearest_idx = idx[:, 1]

    # 确定不属于同一个连通区域的对
    mutual_pairs = []
    for i, nearest in enumerate(nearest_idx):
        pt1 = pts1[i]
        pt2 = pts1[nearest]
        # 检查两点是否属于同一个连通区域
        if labeled_img[pt1[0], pt1[1]] != labeled_img[pt2[0], pt2[1]]:
            mutual_pairs.append((pt1, pt2))

    return mutual_pairs
# 找到互为最近邻的配对
def find_mutual_closest_pairs(pts1, pts2, labeled_img):
    mutual_pairs = []
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    # 对于pts1中的每个点，找到pts2中的最近点
    dists12, idx12 = tree1.query(pts2, k=1)
    # 对于pts2中的每个点，找到pts1中的最近点
    dists21, idx21 = tree2.query(pts1, k=1)

    for i, j in enumerate(idx21):
        # 确保是互相最近的点，并且不属于同一个连通区域
        if idx12[j] == i and labeled_img[pts1[i][0], pts1[i][1]] != labeled_img[pts2[j][0], pts2[j][1]]:
            mutual_pairs.append((pts1[i], pts2[j]))

    return mutual_pairs


# 找到最小连通区域内最近的端点或交叉点
def find_smallest_cc_and_nearest_point(pairs, endpoints_coords, crosspoints_coords, labeled_img):
    smallest_cc_nearest_points = []
    for pt1, pt2 in pairs:
        cc1 = labeled_img[pt1[0], pt1[1]]
        cc2 = labeled_img[pt2[0], pt2[1]]
        smallest_cc = cc1 if np.sum(labeled_img == cc1) < np.sum(labeled_img == cc2) else cc2
        other_point = pt1 if smallest_cc == cc2 else pt2
        cc_points = [pt for pt in np.concatenate((endpoints_coords, crosspoints_coords)) if
                     labeled_img[pt[0], pt[1]] == smallest_cc]
        if not cc_points:
            continue
        tree = cKDTree(cc_points)
        dist, idx = tree.query(other_point, k=1)
        nearest_point = cc_points[idx]
        smallest_cc_nearest_points.append((other_point, nearest_point))
    return smallest_cc_nearest_points


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham 线算法生成器，返回构成线段的所有像素坐标。

    :param x0: 起点 x 坐标
    :param y0: 起点 y 坐标
    :param x1: 终点 x 坐标
    :param y1: 终点 y 坐标
    :return: 构成线段的像素坐标列表
    """
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # error value e_xy
    while True:
        points.append((x0, y0))  # 当前点加入列表
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:  # e_xy+e_x > 0
            err += dy
            x0 += sx
        if e2 <= dx:  # e_xy+e_y < 0
            err += dx
            y0 += sy
    return points


def visualize_and_save_all(skeleton_img, endpoints_coords, crosspoints_coords, end_to_end_pairs, end_to_cross_pairs,
                           save_path):
    # 复制骨架图像用于绘制，保留原图
    visual_img = np.stack((skeleton_img,) * 3, axis=-1) * 255  # 转换为RGB

    # 标记端点，用红色表示
    for y, x in endpoints_coords:
        visual_img[y, x] = [255, 0, 0]  # Red

    # 标记交叉点，用绿色表示
    for y, x in crosspoints_coords:
        visual_img[y, x] = [0, 255, 0]  # Green

    # 绘制端点到端点的配对，用蓝色表示
    # for (y1, x1), (y2, x2) in end_to_end_pairs:
    #     cv2.line(visual_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for (y1, x1), (y2, x2) in end_to_end_pairs:
        # 使用 Bresenham 算法获取线段上的像素坐标
        line_pixels = bresenham_line(x1, y1, x2, y2)

        cv2.line(visual_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        print(line_pixels)
    # 绘制端点到交叉点的配对，也用蓝色表示
    for (y1, x1), (y2, x2) in end_to_cross_pairs:
        cv2.line(visual_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 保存可视化图像
    io.imsave(save_path, visual_img)


# 主逻辑
def main(file_path):
    skeleton_img = load_and_skeletonize_image(file_path)
    endpoints, crosspoints = extract_points(skeleton_img)
    endpoints_coords = points_to_coords(endpoints)
    crosspoints_coords = points_to_coords(crosspoints)
    labeled_skeleton, _ = label(skeleton_img, return_num=True)

    mutual_closest_end_to_end = find_mutual_closest_pairs_v2(endpoints_coords, endpoints_coords, labeled_skeleton)
    mutual_closest_end_to_cross = find_mutual_closest_pairs(endpoints_coords, crosspoints_coords, labeled_skeleton)

    smallest_cc_nearest_end_to_end = find_smallest_cc_and_nearest_point(mutual_closest_end_to_end, endpoints_coords,
                                                                        crosspoints_coords, labeled_skeleton)
    smallest_cc_nearest_end_to_cross = find_smallest_cc_and_nearest_point(mutual_closest_end_to_cross, endpoints_coords,
                                                                          crosspoints_coords, labeled_skeleton)

    filtered_end_to_end = [pair for pair in smallest_cc_nearest_end_to_end if
                           not is_line_crossing_other_cc(pair[0], pair[1], labeled_skeleton)]
    filtered_end_to_cross = [pair for pair in smallest_cc_nearest_end_to_cross if
                             not is_line_crossing_other_cc(pair[0], pair[1], labeled_skeleton)]

    # 使用之前计算的结果进行可视化
    visualize_and_save_all(skeleton_img, endpoints_coords, crosspoints_coords, filtered_end_to_end,
                           filtered_end_to_cross, 'pairs2.png')
    # 先可视化端点到端点的配对

    print(f'端点到端点的配对数: {len(smallest_cc_nearest_end_to_end)}')
    print(f'端点到交叉点的配对数: {len(smallest_cc_nearest_end_to_cross)}')

file_path =r'D:\some CV\try by myself\后处理\cleaned_skeleton.png'
main(file_path)
