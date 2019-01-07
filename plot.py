#!/usr/bin/env python3

import os
import glob
import struct
import numpy as np
import argparse
import matplotlib.pyplot as plt



def load_ndfile(filename):
    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)



def load_checkpoint(btdir):
    database = dict()

    for patch in os.listdir(btdir):

        fd = os.path.join(btdir, patch)
        pd = dict()

        for field in os.listdir(fd):
            fe = os.path.join(fd, field)
            pd[field] = load_ndfile(fe)

        database[patch] = pd

    return database



def imshow_database(database, database1):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    difference = []
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)

    for patch in database:

        R = database[patch]['vert_coords'][:,:,0]
        Q = database[patch]['vert_coords'][:,:,1]
        D = database[patch]['conserved'][:,:,0]
        V = database[patch]['conserved'][:,:,1]

        D1 = database1[patch]['conserved'][:,:,0]

        X = R * np.cos(Q)
        Y = R * np.sin(Q)

        im1 = ax1.pcolormesh(Y, X, D - D1, edgecolor='none', lw=0.5)
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        difference.append(np.linalg.norm(D-D1, ord=2, axis=1)) #error using L2

    ax1.set_title('Log density')
    ax1.set_aspect('equal')
    plt.show()



def hse_diff(database, database1):
    difference = []
    for patch in database:

        R = database[patch]['vert_coords'][:,:,0]
        Q = database[patch]['vert_coords'][:,:,1]
        D = database[patch]['conserved'][:,:,0]
        V = database[patch]['conserved'][:,:,1]

        D1 = database1[patch]['conserved'][:,:,0]

        X = R * np.cos(Q)
        Y = R * np.sin(Q)
        difference.append(np.linalg.norm(D - D1, ord=2)) #error using L2

    scalar_diff = np.linalg.norm(difference, ord=2)
    return scalar_diff


def plot_error_vs_time(filenames, ax1):
    time_series = []
    db = load_checkpoint(filenames[0])
    for dbx in filenames[1:]:
        time_series.append(hse_diff(db, load_checkpoint(dbx)))
    ax1.plot(np.linspace(1, 100, 100), time_series, label=os.path.dirname(filenames[0]))
    ax1.set_title("Time series of HSE L2 error")



def plot_error_vs_resolution(path, ax1):
    for filedir in os.listdir(path):
        chkpts = sorted(glob.glob(path + filedir + '/chkpt.*'))
        plot_error_vs_time(chkpts, ax1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    # plot_error_vs_time(args.filenames, ax1)
    plot_error_vs_resolution("./data/", ax1)
    plt.legend()
    plt.show()
    # db1 = load_checkpoint(args.filenames[1])
    # imshow_database(db, db1)
