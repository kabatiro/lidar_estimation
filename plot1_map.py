from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math


#
#
# Cygnassの重心と1本のレイの交点
#
#

def main(stl, d, fov, ray):
    obj = mesh.Mesh.from_file(stl)
    obj_real = real_scale(obj)
    d_array = np.array(d)
    obj_real.vectors += d_array
    show(obj_real, ray, fov)


def intersection_min(data, ray):  # dataはmesh.Mesh型、dはレイベクトル。
    ans_min = []
    ray = ray / np.linalg.norm(ray)
    t_min = 100000000000
    for vector in data.vectors:
        e1 = vector[1] - vector[0]
        e2 = vector[2] - vector[0]
        r = -vector[0]
        u = np.cross(ray, e2)
        v = np.cross(r, e1)
        if np.inner(u, e1) > 10 ** -3:
            t, beta, ganma = [np.inner(v, e2), np.inner(u, r), np.inner(v, ray)] / (np.inner(u, e1))
            if beta >= 0 and ganma >= 0 and (beta + ganma) <= 1:
                if t_min > t:
                    t_min = t
    ans_min.append(t_min * ray)
    return ans_min


def show(data, ray, fov):  # dataはmesh.Meshオブジェクト,dはレイベクトル。
    fig = plt.figure()
    cog_norm = np.linalg.norm(data.get_mass_properties()[1])
    axes = fig.add_subplot(projection='3d')
    d1 = np.linspace(0, 1.5*ray[0], 2 )
    d2 = np.linspace(0, 1.5*ray[1], 2 )
    d3 = np.linspace(0, 1.5*ray[2], 2 )
    axes.plot(d1, d2, d3, color='r', linewidth=0.5)
    X1 = data.x.flatten()
    Y1 = data.y.flatten()
    Z1 = data.z.flatten()
    axes.plot(X1, Y1, Z1)
    inter = intersection_min(data, ray)
    for i in inter:
        axes.scatter(i[0], i[1], i[2], color='k')
    axes.scatter(data.get_mass_properties()[1][0], data.get_mass_properties()[1][1], data.get_mass_properties()[1][2],
                 marker='*', color='y')
    fov_line = [[], [], []]
    for coor in fov_area(data, fov):
        for j, k in enumerate(coor):
            fov_line[j].append(k)
    for i in range(3):
        fov_line[i].append(fov_line[i][0])
    axes.plot(fov_line[0], fov_line[1], fov_line[2], color='r')
    scale = data.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()


def real_scale(data, x=1635, y=229, z=521):
    x_ratio = x / (data.x.flatten().max() - data.x.flatten().min())
    y_ratio = y / (data.y.flatten().max() - data.y.flatten().min())
    z_ratio = z / (data.z.flatten().max() - data.z.flatten().min())
    data.x = data.x * x_ratio
    data.y = data.y * y_ratio
    data.z = data.z * z_ratio
    data.vectors -= data.get_mass_properties()[1]
    data_real = data
    return data_real


def fov_area(data, fov):
    cog_vertices = []
    cog = data.get_mass_properties()[1]
    cogx = cog[0]
    cogy = cog[1]
    cogz = cog[2]
    cog_size = np.linalg.norm(cog)
    a = 1 + (cogy / cogx) ** 2
    b = - ((cog_size ** 2 - cogz ** 2) * cogy) / (cogx ** 2)
    c = ((cog_size ** 2 - cogz ** 2) ** 2 / cogx ** 2) - (cog_size / math.cos(math.radians(fov / 2))) ** 2 + cogz ** 2
    t = answer2eq(a, b, c)
    cog_mid_z0 = [[(cog_size ** 2 - cogz ** 2 - cogy * t[0]) / cogx, t[0], cogz],
                  [(cog_size ** 2 - cogz ** 2 - cogy * t[1]) / cogx, t[1], cogz]]
    theta = math.atan(cogy / cogx)
    psi = math.atan(cogz / math.sqrt(cogx ** 2 + cogy ** 2))
    for i in cog_mid_z0:
        cog_mid_z0_trans = L_x(-cog_size) * R_y(psi) * R_z(-theta) @ np.append(i, 1)
        cog_vertice1_trans = [[0, cog_mid_z0_trans[0, 1], cog_mid_z0_trans[0, 1]],
                              [0, cog_mid_z0_trans[0, 1], -cog_mid_z0_trans[0, 1]]]
        for j in cog_vertice1_trans:
            cog_vertice_item = R_z(theta) * R_y(-psi) * L_x(cog_size) @ np.append(j, 1)
            cog_vertice_coor = cog_vertice_item.tolist()[0]
            cog_vertices.append([cog_vertice_coor[0], cog_vertice_coor[1], cog_vertice_coor[2]])
    return cog_vertices


def cos(a):
    return math.cos(a)


def sin(a):
    return math.sin(a)


def R_y(a):
    return np.matrix([[cos(a), 0, sin(a), 0], [0, 1, 0, 0], [-sin(a), 0, cos(a), 0], [0, 0, 0, 1]])


def R_z(a):
    return np.matrix([[cos(a), -sin(a), 0, 0], [sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def L_x(a):
    return np.matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def answer2eq(a, b, c):
    ans = [(- b - math.sqrt(b ** 2 - a * c)) / a, (- b + math.sqrt(b ** 2 - a * c)) / a]
    return ans


if __name__ == '__main__':
    main('cygnss.stl', [1000, 5000, 1000], 20, [1000, 5000, 1000])

