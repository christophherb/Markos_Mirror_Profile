import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

from numpy import sin, cos, cosh

def rotate_array(xyz, alpha, beta, gamma):
    """takes an array (x, y, z) and a set of rotation angles, alpha, beta, gamma and reports

    Args:
        xyz (array): an array with x, y, z values
        alpha (float): angle around x
        beta (float): angle around y
        gamma (float): angle around z

    Returns:
        xs, ys, zs: rotated xs 
    """
    rot_x = []
    rot_y = []
    rot_z = []
    for ind, val in enumerate(xyz):
        x, y, z = val
        rot_vec = rotation_euler(x, y, z, alpha, beta, gamma)
        rot_x.append(rot_vec[0])
        rot_y.append(rot_vec[1])
        rot_z.append(rot_vec[2])
    
    return np.array(rot_x), np.array(rot_y), np.array(rot_z)

def cylindric_fit(y, y0, b, a, z0):
    return -b*(1-(y-y0)**2/a**2)**0.5+z0

def parabolic_fit(y, a, y0, c):
    """for a given x and y (x is ignored here), returns the elevation z = a*(y-y0)**2+c

    Args:
        x (float): coordinate along which the shape is translationally invariant
        y (float): coordinate on which the elevation depends
        a (float): prefactor of the parabola
        y0 (float): center of the parabola in y
        c (float): offset of the parabola
    """
    return c+ a*(y-y0)**2+c

def cosh_fit(y, a, A, y0, c, alpha=0, beta=0, gamma=0):
    """returns the cosh_fit

    Args:
        y (_type_): _description_
        a (_type_): _description_
        A (_type_): _description_
        y0 (_type_): _description_
        c (_type_): _description_
        alpha (int, optional): _description_. Defaults to 0.
        beta (int, optional): _description_. Defaults to 0.
        gamma (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return A*(cosh(a*(y-y0))-1)+c

def rotation_euler(x, y, z, alpha, beta, gamma):
    """rotates the values x y and z around angles alpha beta gamma respectively

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinat
        alpha (float): angle around x
        beta (float): angles around y
        gamma (float): angle around z

    Returns:
        np.array: rotated vector x, y, z
    """
    mat = np.array([
        [cos(beta)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma), cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],
        [cos(beta)*sin(gamma), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma)],
        [-sin(beta), sin(alpha)*cos(beta), cos(alpha)*cos(beta)]
    ])#matrix of rotation as copied from wikipedia
    vec = np.array([
        [x],
        [y],
        [z]
    ])
    return mat @ vec

def zero_res_parabolic(xyz, a, y0, c, alpha, beta, gamma):
    """parabolic fit with rotated values

    Args:
        xyz (_type_): _description_
        a (_type_): _description_
        y0 (_type_): _description_
        c (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_y = np.asarray(rot_y)
    rot_z = np.asarray(rot_z)
    calc_z = parabolic_fit(rot_y, a, y0, c)
    return calc_z-rot_z

def zero_res_cosh(xyz, a, A, y0, c, alpha, beta, gamma):
    """cosh fit with rotated values

    Args:
        xyz (_type_): _description_
        a (_type_): _description_
        A (_type_): _description_
        y0 (_type_): _description_
        c (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_z = np.asarray(rot_z)
    calc_z = cosh_fit(rot_y, a, A, y0, c)
    return calc_z-rot_z

def zero_res_cyl(xyz, y0, b, a, z0, alpha, beta, gamma):
    """cylinder fit with rotated values

    Args:
        xyz (_type_): _description_
        y0 (_type_): _description_
        b (_type_): _description_
        a (_type_): _description_
        r (_type_): _description_
        z0 (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_y = np.asarray(rot_y)
    rot_z = np.asarray(rot_z)
    calc_z = cylindric_fit(rot_y, y0, b, a, z0)
    return calc_z-rot_z

def return_parab_val(xyz, a, y0, c, alpha, beta, gamma):
    """parabolic fit with rotated values

    Args:
        xyz (_type_): _description_
        a (_type_): _description_
        y0 (_type_): _description_
        c (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_y = np.asarray(rot_y)
    rot_z = np.asarray(rot_z)
    calc_z = parabolic_fit(rot_y, a, y0, c)
    return rot_x, rot_y, rot_z, calc_z

def return_cosh_val(xyz, a, A, y0, c, alpha, beta, gamma):
    """cosh fit with rotated values

    Args:
        xyz (_type_): _description_
        a (_type_): _description_
        A (_type_): _description_
        y0 (_type_): _description_
        c (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_z = np.asarray(rot_z)
    calc_z = cosh_fit(rot_y, a, A, y0, c)
    return rot_x, rot_y, rot_z, calc_z

def return_cyl_val(xyz, y0, b, a, z0, alpha, beta, gamma):
    """cylinder fit with rotated values

    Args:
        xyz (_type_): _description_
        y0 (_type_): _description_
        b (_type_): _description_
        a (_type_): _description_
        r (_type_): _description_
        z0 (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    rot_x, rot_y, rot_z = rotate_array(xyz, alpha, beta, gamma)
    rot_y = np.asarray(rot_y)
    rot_z = np.asarray(rot_z)
    calc_z = cylindric_fit(rot_y, y0, b, a, z0)
    return rot_x, rot_y, rot_z, calc_z
