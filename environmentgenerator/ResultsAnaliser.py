# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:08:53 2022

@author: migue
"""
from glob import glob
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from PIL import Image
from matplotlib import cm
import cv2
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from mayavi import mlab

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    'font.size': 55
})


def sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def ravzip(*itr):
    '''flatten and zip arrays'''
    return zip(*map(np.ravel, itr))


def getDistances(X, Y, camera_pos):
    xyz_data = np.stack([X, Y, np.zeros_like(X)])
    return np.linalg.norm(xyz_data - camera_pos[:, None, None], axis=0)


def load_OBJ(obj_path):
    collection = np.load(obj_path)
    if collection.ndim == 4:
        if collection[0].shape[0] == 4:
            return [np.transpose(obj, (1, 2, 0)) for obj in collection]
        else:
            return [obj for obj in collection]
    elif collection.ndim == 3:
        if collection.shape[0] == 4:
            return [np.transpose(collection, (1, 2, 0))]
        else:
            return [collection]
    else:
        print("Unexpected shape: " + str(collection.shape))


## matches between variables in our code and sample
# Y == yposM
# X == xposM
# dx, dy are calculated width and height of the bars so that the bars exactly fit the space (we don't care, set both to 0.5)
# Zg == data (AS IS)


def plot_similarity_matrix(data, mapping):
    fig, ax1 = plt.subplots(figsize=(16, 16), dpi=300)
    ax1.set_xlabel('Synthetic instances', labelpad=20)
    ax1.set_ylabel('Real instances', labelpad=20)

    # scalar values (1D)
    xpos = np.arange(data.shape[0])
    ypos = np.arange(data.shape[1])
    # zpos = data.ravel()

    # matrix values (2D)
    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
    xposM = xposM.astype(np.float32)
    yposM = yposM.astype(np.float32)

    # hide tick labels on the axes
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

    # compute color values
    max_height = data.max()
    min_height = data.min()
    cmap = cm.get_cmap('RdYlGn')
    if mapping == 'direct':
        rgba = cmap((data - min_height) / max_height)
    elif mapping == 'inverse':
        rgba = cmap(1 - (data - min_height) / max_height)
        cmap = cm.get_cmap('RdYlGn_r')

    ax1.matshow(rgba)
    plt.colorbar(cm.ScalarMappable(cmap=cmap))
    plt.show()


def plot_similarity(data, mapping):
    fig = plt.figure(figsize=(16, 16), dpi=300)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Synthetic instances', labelpad=20)
    ax1.set_ylabel('Real instances', labelpad=20)
    # print(ax1.xaxis._axinfo['tick'])
    # ax1.set_zlabel('Thickness nm')
    xpos = np.arange(data.shape[0])
    ypos = np.arange(data.shape[1])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = data.ravel()

    # dx = 0
    # dy = 0
    # dz = zpos+0.5
    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(axis='z', which='major', pad=20)

    # values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    # colors = cm.rainbow(values)
    cmap = cm.get_cmap('RdYlGn')
    max_height = np.max(dz)  #
    min_height = np.min(dz)
    if mapping == 'direct':
        rgba = [cmap((k - min_height) / max_height) for k in dz]
    elif mapping == 'inverse':
        rgba = [cmap(1 - (k - min_height) / max_height) for k in dz]
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=rgba)
    plt.show()


def plot_similarity_2(data, mapping):
    xpos = np.arange(data.shape[0])
    ypos = np.arange(data.shape[1])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = data.ravel()

    # dx = 0
    # dy = 0
    # dz = zpos+0.5
    dx = 0.5
    dy = 0.5
    dz = zpos

    max_height = np.max(dz)  #
    min_height = np.min(dz)
    mlab.axes(extent=[1, 36, 1, 36, 0, 1])
    mlab.barchart(xposM.ravel(), yposM.ravel(), dz * 100)


def plot_similarity_3(data, mapping):
    fig, ax1 = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(16, 16), dpi=300)
    ax1.set_xlabel('Synthetic instances', labelpad=20)
    ax1.set_ylabel('Real instances', labelpad=20)

    # scalar values (1D)
    xpos = np.arange(data.shape[0])
    ypos = np.arange(data.shape[1])
    zpos = data.ravel()

    # matrix values (2D)
    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
    xposM = xposM.astype(np.float32)
    yposM = yposM.astype(np.float32)
    zposM = data

    # bar sizes (base shape of the column)
    dx = 0.5
    dy = 0.5
    # adapt x & y so that the bar's center is on an integer value
    # xposM -= dx
    # yposM -= dy

    # show ticks on the axes
    ax1.w_xaxis.set_ticks(xpos)
    ax1.w_yaxis.set_ticks(ypos)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(axis='z', which='major', pad=20)

    # compute color values
    cmap = cm.get_cmap('RdYlGn')
    max_height = zposM.max()
    min_height = zposM.min()
    if mapping == 'direct':
        rgba = np.asarray([cmap((k - min_height) / max_height) for k in zpos]).reshape([37, 37, 4])
    elif mapping == 'inverse':
        rgba = np.asarray([cmap(1 - (k - min_height) / max_height) for k in zpos]).reshape([37, 37, 4])

    # Compute depth values of bars to render them in the right order
    x1, y1, z1 = sph2cart(*sphview(ax1))
    camera = np.array((x1, y1, 0))
    zo = getDistances(xposM, yposM, camera)
    zo = zo.max() - zo

    # xyz = np.array(sph2cart(*sphview(ax1)), ndmin=3).T #camera position in xyz
    # zo = np.multiply([xposM, yposM, np.zeros_like(zposM)], xyz).sum(0)  #"distance" of bars from camera

    bars = np.empty(xposM.shape, dtype=object)
    for i, (x, y, dz, o) in enumerate(ravzip(xposM, yposM, zposM, zo)):
        j, k = divmod(i, 37)
        bars[j, k] = pl = ax1.bar3d(x, y, 0, dx, dy, dz, color=rgba[j, k], zsort='min')
        pl._sort_zpos = o
    plt.show()


def plot_similarity_3d(data):
    z = data
    nrows, ncols = data.shape
    # x = np.linspace(dem['xmin'], dem['xmax'], ncols)
    # y = np.linspace(dem['ymin'], dem['ymax'], nrows)
    x = range(nrows)
    y = range(ncols)
    x, y = np.meshgrid(x, y)

    region = np.s_[5:50, 5:50]
    x, y, z = x[region], y[region], z[region]

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    plt.show()


def load_results():
    cs = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\cs.npy")
    ergas = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\ergas.npy")
    mse = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\mse.npy")
    mssim = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\mssim.npy")
    psnr = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\psnr.npy")
    rase = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\rase.npy")
    rmse = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\rmse.npy")
    sam = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\sam.npy")
    scc = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\scc.npy")
    ssim = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\ssim.npy")
    uqi = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\uqi.npy")
    vipf = np.load(r"F:\Storage\Master Thesis - Storage\Analysis\vipf.npy")
    return cs, ergas, mse, mssim, psnr, rase, rmse, sam, scc, ssim, uqi, vipf


# https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6

def load_data():
    path_real = r'F:\Storage\Master Thesis - Storage\Blender Processor Results\Training data - selected OBJs - Light reduced2021-05-25 19h25m45s\preprocessed_global_average_empty=0_res=512\normalised'
    # path_fake = r'F:\Storage\Master Thesis - Storage\SGAN Generated\128'
    path_fake = r'F:\Storage\Master Thesis - Storage\SGAN Generated\512'

    path_real = Path(path_real)
    path_fake = Path(path_fake)

    print("Searching for Numpy files in input paths...")
    files_real = [file for file in path_real.glob("*.npy")]
    files_fake = [file for file in path_fake.glob("*.npy")]

    array_real = sum(Parallel(n_jobs=-1, backend="loky")(
        map(delayed(load_OBJ),
            (obj_path for obj_path in tqdm(files_real, desc="Loading array A")))), [])
    array_fake = sum(Parallel(n_jobs=-1, backend="loky")(
        map(delayed(load_OBJ),
            (obj_path for obj_path in tqdm(files_fake, desc="Loading array B")))), [])
    return array_real, array_fake

    ######################################### HISTOGRAMS


def plot_histograms(array_real, array_fake):
    ######## DEPTH ########
    height_values_real = np.concatenate([file[:, :, 3].flatten() for file in array_real])
    height_values_real = 2 * (height_values_real - height_values_real.min()) / (
                height_values_real.max() - height_values_real.min()) - 1

    height_values_fake = -np.concatenate([file[:, :, 3].flatten() for file in array_fake])

    bins = np.linspace(-1, 1, 100)
    plt.hist(height_values_real, bins, alpha=0.5, label='Training set')
    plt.hist(height_values_fake, bins, alpha=0.5, label='Artificial set')
    plt.legend(loc='upper right')
    plt.show()

    all_rgb = np.concatenate([file[:, :, :3].flatten() for file in array_real])
    ######## RED ########
    red_values_real = np.concatenate([file[:, :, 0].flatten() for file in array_real])
    red_values_real = 2 * (red_values_real - all_rgb.min()) / (all_rgb.max() - all_rgb.min()) - 1
    red_values_fake = np.concatenate([file[:, :, 0].flatten() for file in array_fake])
    bins = np.linspace(-1, 1, 100)
    plt.hist(red_values_real, bins, alpha=0.5, label='Training set')
    plt.hist(red_values_fake, bins, alpha=0.5, label='Artificial set')
    plt.legend(loc='upper right')
    plt.show()
    ######## GREEN ########
    green_values_real = np.concatenate([file[:, :, 1].flatten() for file in array_real])
    green_values_real = 2 * (green_values_real - all_rgb.min()) / (all_rgb.max() - all_rgb.min()) - 1
    green_values_fake = np.concatenate([file[:, :, 1].flatten() for file in array_fake])
    bins = np.linspace(-1, 1, 100)
    plt.hist(green_values_real, bins, alpha=0.5, label='Training set')
    plt.hist(green_values_fake, bins, alpha=0.5, label='Artificial set')
    plt.legend(loc='upper right')
    plt.show()
    ######## BLUE ########
    blue_values_real = np.concatenate([file[:, :, 2].flatten() for file in array_real])
    blue_values_real = 2 * (blue_values_real - all_rgb.min()) / (all_rgb.max() - all_rgb.min()) - 1
    blue_values_fake = np.concatenate([file[:, :, 2].flatten() for file in array_fake])
    bins = np.linspace(-1, 1, 100)
    plt.hist(blue_values_real, bins, alpha=0.5, label='Training set')
    plt.hist(blue_values_fake, bins, alpha=0.5, label='Artificial set')
    plt.legend(loc='upper right')
    plt.show()


#########################################  IMAGE SIMILARITY
def calculate_similarity(array_real, array_fake):
    # measures
    dimension = min(len(array_real), len(array_fake))
    mse_measures = np.empty([dimension, dimension])
    rmse_measures = np.empty([dimension, dimension])
    psnr_measures = np.empty([dimension, dimension])
    ssim_measures = np.empty([dimension, dimension])
    cs_measures = np.empty([dimension, dimension])  # has to do with ssim. "compressed sensing"?
    uqi_measures = np.empty([dimension, dimension])
    msssim_measures = np.empty([dimension, dimension])
    ergas_measures = np.empty([dimension, dimension])
    scc_measures = np.empty([dimension, dimension])
    rase_measures = np.empty([dimension, dimension])
    sam_measures = np.empty([dimension, dimension])
    vifp_measures = np.empty([dimension, dimension])
    # normalised images
    fake_norm = np.asarray(
        [cv2.normalize(landscape, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for landscape in
         np.asarray(array_fake)], np.uint8)
    real_norm = np.asarray(
        [cv2.normalize(np.asarray(landscape), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for
         landscape in np.asarray(array_real)], np.uint8)

    # for i, fake_env in tqdm(enumerate(array_fake), "Calculating image similarity measures"):
    for i in tqdm(range(dimension), "Calculating image similarity measures"):
        # rgbd_fake = Image.fromarray(array_fake[i], 'RGBA')
        # rgbd_fake = Image.fromarray(fake_env, 'RGBA')
        # print(fake_env.shape)
        # print(rgbd_fake.size)
        # for j, real_env in enumerate(array_real):
        for j in range(dimension):
            # rgbd_real = Image.fromarray(array_real[j], 'RGBA')
            # rgbd_real = Image.fromarray(real_env, 'RGBA')
            # print(real_env.shape)
            # print(rgbd_real.size)
            print("Doing mse: [{},{}]".format(i, j))
            mse_measures[i, j] = mse(array_fake[i], array_real[j])
            print("Doing rmse: [{},{}]".format(i, j))
            rmse_measures[i, j] = rmse(array_fake[i], array_real[j])
            print("Doing psnr: [{},{}]".format(i, j))
            psnr_measures[i, j] = psnr(fake_norm[i], real_norm[j])
            print("Doing ssim, cs: [{},{}]".format(i, j))
            ssim_measures[i, j], cs_measures[i, j] = ssim(fake_norm[i], real_norm[j])
            print("Doing uqi: [{},{}]".format(i, j))
            uqi_measures[i, j] = uqi(array_fake[i], array_real[j])
            print("Doing msssim: [{},{}]".format(i, j))
            msssim_measures[i, j] = msssim(fake_norm[i], real_norm[j])
            print("Doing ergas: [{},{}]".format(i, j))
            ergas_measures[i, j] = ergas(array_fake[i], array_real[j])
            print("Doing scc: [{},{}]".format(i, j))
            scc_measures[i, j] = scc(array_fake[i], array_real[j])
            print("Doing rase: [{},{}]".format(i, j))
            rase_measures[i, j] = rase(array_fake[i], array_real[j])
            print("Doing sam: [{},{}]".format(i, j))
            sam_measures[i, j] = sam(array_fake[i], array_real[j])
            print("Doing vifp: [{},{}]".format(i, j))
            vifp_measures[i, j] = vifp(array_fake[i], array_real[j])


cs_measures, ergas_measures, mse_measures, mssim_measures, psnr_measures, rase_measures, rmse_measures, sam_measures, scc_measures, ssim_measures, uqi_measures, vifp_measures = load_results()
dimension = min(cs_measures.shape[0], cs_measures.shape[0])

plot_similarity(mse_measures)
plot_similarity(rmse_measures)
plot_similarity(psnr_measures)
plot_similarity(ssim_measures)
plot_similarity(uqi_measures)
plot_similarity(mssim_measures)
plot_similarity(ergas_measures)
plot_similarity(scc_measures)
plot_similarity(rase_measures)
plot_similarity(sam_measures)
plot_similarity(vifp_measures)