from typing import List, Any
from mpl_toolkits import mplot3d
from stl import mesh
import matplotlib.pyplot as plt
import numpy as np
import math
import time


#
#
# Cygnassの重心と1本のレイの交点
#
#

def main(stl, d, fov, ray_center, res):
    data_real = real_scale(stl, d)
    show(data_real, fov, ray_center, res)


def intersection_min(data, ray):  # dataはmesh.Mesh型、dはレイベクトル。
    ans_min = []
    ray = ray / np.linalg.norm(ray)
    t_min = 10000000
    for vector in data.vectors:
        e1 = vector[1] - vector[0]
        e2 = vector[2] - vector[0]
        r = -vector[0]
        u = np.cross(ray, e2)
        v = np.cross(r, e1)
        if np.inner(u, e1) > 10 ** -3:
            beta = np.inner(u, r) / np.inner(u, e1)
            if beta >= 0:
                ganma = np.inner(v, ray) / np.inner(u, e1)
                if ganma >= 0 and (beta + ganma) <= 1:
                    t = np.inner(v, e2) / np.inner(u, e1)
                    if t_min > t:
                        t_min = t
    if not 10000000 == t_min:
        ans_min.append(t_min * ray)
    return ans_min


def show(data, fov, ray_center, res):  # dataはmesh.Meshオブジェクト,dはレイベクトル。
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    X1 = data.x.flatten()
    Y1 = data.y.flatten()
    Z1 = data.z.flatten()
    axes.plot(X1, Y1, Z1)
    dot = rays_func(data, fov, ray_center, res)
    t0 = time.time()
    for ray in dot:
        d1 = np.linspace(0, 1.5 * ray[0], 2)
        d2 = np.linspace(0, 1.5 * ray[1], 2)
        d3 = np.linspace(0, 1.5 * ray[2], 2)
        axes.plot(d1, d2, d3, color='r', linewidth=0.5)

        inter = intersection_min(data, ray)
        for i in inter:
            axes.scatter(i[0], i[1], i[2], color='k')
        axes.scatter(data.get_mass_properties()[1][0],
                     data.get_mass_properties()[1][1],
                     data.get_mass_properties()[1][2],
                     marker='*', color='y')
    fov_line = [[], [], []]
    print("calculate intersection:{}".format(time.time() - t0) + "[sec]")
    for coor in fov_area(data, fov, ray_center):
        for j, k in enumerate(coor):
            fov_line[j].append(k)
    for i in range(3):
        fov_line[i].append(fov_line[i][0])
    axes.plot(fov_line[0], fov_line[1], fov_line[2], color='k')
    scale = data.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()


def real_scale(stl, d, x=1635, y=229, z=521):
    data = mesh.Mesh.from_file(stl)
    x_ratio = x / (data.x.flatten().max() - data.x.flatten().min())
    y_ratio = y / (data.y.flatten().max() - data.y.flatten().min())
    z_ratio = z / (data.z.flatten().max() - data.z.flatten().min())
    data.x = data.x * x_ratio
    data.y = data.y * y_ratio
    data.z = data.z * z_ratio
    data.vectors -= data.get_mass_properties()[1]
    d_array = np.array(d)
    data.vectors += d_array
    return data


def fov_area(data, fov, ray_center):
    cog_vertices: List[List[Any]] = []
    ray_center_norm = np.linalg.norm(ray_center)
    cog_size = np.linalg.norm(data.get_mass_properties()[1])
    ray_center_x = ray_center[0] * cog_size / ray_center_norm
    ray_center_y = ray_center[1] * cog_size / ray_center_norm
    ray_center_z = ray_center[2] * cog_size / ray_center_norm
    a = 1 + (ray_center_y / ray_center_x) ** 2
    b = - ((cog_size ** 2 - ray_center_z ** 2) * ray_center_y) / (ray_center_x ** 2)
    c = ((cog_size ** 2 - ray_center_z ** 2) ** 2 / ray_center_x ** 2) - (
            cog_size / math.cos(math.radians(fov / 2))) ** 2 + ray_center_z ** 2
    t = answer2eq(a, b, c)
    cog_mid_z0 = [[(cog_size ** 2 - ray_center_z ** 2 - ray_center_y * t[0]) / ray_center_x, t[0], ray_center_z],
                  [(cog_size ** 2 - ray_center_z ** 2 - ray_center_y * t[1]) / ray_center_x, t[1], ray_center_z]]
    theta = math.atan(ray_center_y / ray_center_x)
    psi = math.atan(ray_center_z / math.sqrt(ray_center_x ** 2 + ray_center_y ** 2))
    for i in cog_mid_z0:
        cog_mid_z0_trans = L_x(-cog_size) @ R_y(psi) @ R_z(-theta) @ np.append(i, 1)
        cog_vertice1_trans = np.array([[0, cog_mid_z0_trans[1], cog_mid_z0_trans[1]],
                                       [0, cog_mid_z0_trans[1], -cog_mid_z0_trans[1]]])
        for j in cog_vertice1_trans:
            cog_vertice_item = R_z(theta) @ R_y(-psi) @ L_x(cog_size) @ np.append(j, 1)
            cog_vertice_coor = cog_vertice_item.tolist()
            cog_vertices.append([cog_vertice_coor[0], cog_vertice_coor[1], cog_vertice_coor[2]])
    return cog_vertices


def rays_func(data, fov, ray_center, res):
    v4 = fov_area(data, fov, ray_center)
    side_vectors = []
    side_vectors_unit = []
    dot = []
    for i in range(2):
        side_vectors.append(np.array(v4[i + 1]) - np.array(v4[i]))
    for i in range(2):
        side_vectors_unit.append(side_vectors[i] / res)
    x = side_vectors_unit[0]
    y = side_vectors_unit[1]
    for i in range(res + 1):
        for j in range(res + 1):
            dot.append((x * i + y * j) + np.array(v4[0]))
    return dot


def cos(a):
    return math.cos(a)


def sin(a):
    return math.sin(a)


def R_y(a):
    return np.array([[cos(a), 0, sin(a), 0], [0, 1, 0, 0], [-sin(a), 0, cos(a), 0], [0, 0, 0, 1]])


def R_z(a):
    return np.array([[cos(a), -sin(a), 0, 0], [sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def L_x(a):
    return np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def answer2eq(a, b, c):
    ans = [(- b - math.sqrt(b ** 2 - a * c)) / a, (- b + math.sqrt(b ** 2 - a * c)) / a]
    return ans


if __name__ == '__main__':
    start = time.time()
    main('cygnss.stl', [1000, 1000, 1000], 20, [1000, 1000, 1000], 10)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
