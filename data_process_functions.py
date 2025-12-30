#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:21:49 2025

@author: santapile
"""

import numpy as np
import os
from functools import partial
from scipy.optimize import curve_fit

from scipy.constants import mu_0
from scipy.constants import physical_constants
import matplotlib
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# from . import check_file_and_folder_structure as cffs


matplotlib.rcParams.update({"font.size": 14})


# Kuz'min function
def spontaneous_magnetization(
    T: np.ndarray, M_0: float, s: float, T_C: float, p: float, beta: float
) -> np.ndarray:
    """Formula 4 from article "Exchange stiffness of ferromagnets" by Kuz’min et al.
    See: https://link.springer.com/article/10.1140/epjp/s13360-020-00294-y
    """
    return M_0 * (1 - s * (T / T_C) ** (3 / 2) - (1 - s) * (T / T_C) ** p) ** beta


# get M(T) data from h5 file - used
def get_M_T_data_from_h5(h5_filename):
    data = h5py.File(h5_filename, 'r')['raw_data/MC/M(T)']
    dataStr = str(data[()])[:-1]
    number_cols = len(dataStr.split(sep='\\n')[2].replace('  ', ' ').split(sep=' '))
    # print(number_cols)
    if number_cols == 10:
        columns = ['T', 'M', 'M2', 'M4', 'U_Binder', 'chi', 'C_v', 'E', 'E_exc', 'E_lsf']
    elif number_cols == 11:
        columns = ['T', 'M', 'M2', 'M4', 'U_Binder', 'chi', 'C_v', 'E', 'E_exc', 'E_lsf', 'Cv_new']
    M_T_data = pd.DataFrame(columns=columns)

    for item in dataStr.split(sep='\\n')[1:]:
        item = item.replace('  ', ' ')
        # print(item)
        if len(item.split(sep=' ')) == len(columns):
            M_T_data = pd.concat([M_T_data, pd.DataFrame([[float(x) for x in item.split(sep=' ')]], columns=columns)], ignore_index=True)

    return M_T_data


# get number of atoms per cell from h5 file
def get_n_atoms_per_cell_from_h5(h5_filename):
    momfileData = h5py.File(h5_filename, 'r')['raw_data/MC/momfile']
    return len([item for item in str(momfileData[()]).split(sep='\\n') if len(item) > 1])


# get unit cell volume from h5 file
def get_unit_cell_volume_h5(h5_filename):
    """
    Extracts the unit cell volume from the given file.
    :param file_name: The name of the file to extract the unit cell volume from.
    :return: The unit cell volume in cubic angstroms (A^3).
    """
    out_last = h5py.File(h5_filename, 'r')['raw_data/GS/x/out_last']
    ucv = find_line_out_last(out_last, "unit cell volume:")
    ucvA = ucv[list(ucv.keys())[0]][0] / 1.8897259**3  # unit cell volume in A^3
    return ucvA


# find line in out_last h5 dataset
def find_line_out_last(out_last, valname, printlines=False):
    """
    Find last line in lines (list) with valname (string) and
    return a dict of IDs and valuse coming after valname
    """
    # Check if file exists
    lines = str(out_last[()]).split(sep='\\n')

    if printlines:
        print(lines)

    last_lines_with_valname = {}
    for line in lines:
        if valname in line:
            if printlines:
                print(line)
            pos = line.find(valname) + len(valname)
            # TODO: check if part of the line can be converted to float;
            # introduce boundaries in which the value should be
            key, value = line.split()[0], [float(x) for x in line[pos:].split()]
            last_lines_with_valname[key] = value
    return last_lines_with_valname


# compute anisotropy constant from h5 files
def compute_anisotropy_constant_h5(xyz_dirs, ucvA):
    """
    Docstring for compute_anisotropy_constant_h5
    
    :param xyz_dirs: Description
    :param ucvA: Description
    """
    energies = {}

    for key in list(xyz_dirs.keys()):
        out_MF = xyz_dirs[key]['out_MF_'+key]
        eigenvalue_sum = find_line_out_last(out_MF, "Eigenvalue sum:")
        energies[key] = eigenvalue_sum[list(eigenvalue_sum.keys())[0]][0]

    allKs = list()
    if "z" in energies.keys():
        if "x" in energies.keys():
            Kxz = (energies["x"] - energies["z"]) / ucvA * 2179874
            allKs.append(Kxz)
        if "y" in energies.keys():
            Kyz = (energies["y"] - energies["z"]) / ucvA * 2179874
            allKs.append(Kyz)

    K1_in_JPerCubibm = (
        max(allKs) * 1e6
    )  # anisotropy J/m³; MagnetocrystallineAnisotropyConstantK1

    return K1_in_JPerCubibm


# Kuzmin fit function
def Kuzmin_fit(TK, Js, Tmeas, K1_in_JPerCubibm):
    """
    Fit Kuzmin function to M(T) data and compute magnetic properties at Tmeas K.
    Parameters
    ----------
    TK : np.ndarray
        Array of temperatures (K).
    Js : np.ndarray
        Array of magnetic polarizations (T).
    Tmeas : float
        Measurement temperature (K).
    K1_in_JPerCubibm : float
        Anisotropy constant at 0 K (J/m^3).
    Returns
    -------
    Js_meas : float
        Magnetic polarization at Tmeas K (T).
    A_meas : float
        Exchange constant at Tmeas K (J/m).
    K_meas : float
        Anisotropy constant at Tmeas K (J/m^3).
    xfine : np.ndarray
        Fine temperature grid for plotting (K).
    Js_0 : float
        Magnetic polarization at 0 K (T).
    s : float
        Fitting parameter.
    m_s : callable
        Spontaneous magnetization function.
    """
    poscut = np.argmin(np.diff(Js) / np.diff(TK)) + 2
    try:
        Tc = TK[poscut]
    except IndexError:
        Tc = TK[-1]
        poscut = len(TK) - 1
    print(f"Tc = {Tc} K")
    TKc = TK[:poscut].copy()
    Jsc = Js[:poscut].copy()

    xfine = np.linspace(0, Tc, 500)
    p = 5.0 / 2
    beta = 1.0 / 3
    m_s = partial(spontaneous_magnetization, p=p, beta=beta, T_C=Tc)

    popt, pcov = curve_fit(m_s, TKc, Jsc)
    Js_0, s = popt
    print(Js_0, s)
    # T_fit = np.linspace(min(TKc), max(TKc), 500)
    # Js_fit = m_s(T_fit, Js_0, s)

    g = 2
    k_b = physical_constants["Boltzmann constant"][0]
    mu_b = physical_constants["Bohr magneton"][0]

    M_0 = Js_0 / mu_0
    D = 0.1509 * ((g * mu_b) / (s * beta * M_0)) ** (2.0 / 3) * k_b * Tc
    print("Spin wave stiffness constant ", D)
    A_0 = M_0 * D / (2 * g * mu_b)
    print("Exchange constant A at T=0 (J/m) : ", A_0)

    # Magnetic polarization at Tmeas K
    Js_meas = m_s(Tmeas, Js_0, s)
    print(f"Js_{Tmeas} (T): ", Js_meas)

    A_meas = A_0 * (Js_meas / Js_0) ** 2
    print(f"A_{Tmeas} (J/m): ", A_meas)

    K_meas = K1_in_JPerCubibm * (Js_meas / Js_0) ** 3
    print(f"K_{Tmeas} (J/m^3): ", K_meas)

    return Js_meas, A_meas, K_meas, xfine, Js_0, s, m_s