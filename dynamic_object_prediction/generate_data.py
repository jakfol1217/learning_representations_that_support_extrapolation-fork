#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as math
from matplotlib.path import Path
import os


def make_polygon(n, numEdge, r, rot, xoff, yoff, intensity, distortion, r_mod):
    # Makes single polygon

    # Initialize grid space
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y = x.flatten(), y.flatten()
    grid_points = np.vstack((x, y)).T

    # Initialize array for vertices
    points = []
    current_edge = 0  # parameter rmod re-scales the radius of every vertex to increase irregularity

    # Get vertices of polygon
    for vertex in np.arange(numEdge) + distortion:

        points.append(
            [
                xoff
                + r_mod[current_edge]
                * r
                * math.cos(((rot + vertex / numEdge) * 2 * math.pi)),
                yoff
                + r_mod[current_edge]
                * r
                * math.sin(((rot + vertex / numEdge) * 2 * math.pi)),
            ]
        )
        current_edge += 1

    # Create binary mask for polygon by checking if it contains each grid point
    p = Path(points)
    grid = p.contains_points(grid_points)
    mask = intensity * (grid.reshape(n, n))
    # mask[mask==0] = np.random.rand(1)

    return mask


def make_polygon_sequence(
    n,
    numEdge,
    line,
    r,
    rot,
    xoff,
    yoff,
    intensity,
    sequence_length,
    distort,
    outline,
    noise,
    rmod,
):
    # returns sequence of polygons in one 3D matrix. Can vary along any specified dimension.
    # r, rot, xoff, yoff, intensity should be inputted as 2-value array. If same, will not vary.

    sequence = np.empty([sequence_length, n, n])
    distortion = np.random.uniform(-distort, distort, numEdge)
    gaussian_noise = noise * np.random.randn(sequence_length, n, n)
    r_mod = np.random.uniform(rmod[0], rmod[1], numEdge)

    r = np.linspace(r[0], r[1], num=sequence_length)
    rot = np.linspace(rot[0], rot[1], num=sequence_length)
    xoff = np.linspace(xoff[0], xoff[1], num=sequence_length)
    yoff = np.linspace(yoff[0], yoff[1], num=sequence_length)
    intensity = np.linspace(intensity[0], intensity[1], num=sequence_length)

    for i in range(sequence_length):

        if outline == True:
            outer = make_polygon(
                n,
                numEdge,
                r[i],
                rot[i],
                xoff[i],
                yoff[i],
                intensity[i],
                distortion,
                r_mod,
            )
            inner = make_polygon(
                n,
                numEdge,
                line * r[i],
                rot[i],
                xoff[i],
                yoff[i],
                intensity[i],
                distortion,
                r_mod,
            )
            sequence[i, :, :] = outer - inner
        else:
            sequence[i, :, :] = make_polygon(
                n,
                numEdge,
                r[i],
                rot[i],
                xoff[i],
                yoff[i],
                intensity[i],
                distortion,
                r_mod,
            )

    seq = sequence + gaussian_noise
    normed = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    return normed


def make_polygon_batch(
    num_ex,
    n,
    sequence_length,
    numEdge_vec,
    distort_vec,
    outline_vec,
    r1_vec,
    r2_vec,
    rot1_vec,
    rot2_vec,
    xoff1_vec,
    xoff2_vec,
    yoff1_vec,
    yoff2_vec,
    intensity1_vec,
    intensity2_vec,
    outline,
    noise_vec,
    rmod,
):
    # Creates a training set of sequences

    train_set = np.empty([num_ex, sequence_length, n, n])

    for i in range(num_ex):

        noise = noise_vec[i]
        numEdge = numEdge_vec[i]
        distort = distort_vec[i]
        line = outline_vec[i]
        r = np.array([r1_vec[i], r2_vec[i]])
        rot = np.array([rot1_vec[i], rot2_vec[i]])
        xoff = np.array([xoff1_vec[i], xoff2_vec[i]])
        yoff = np.array([yoff1_vec[i], yoff2_vec[i]])
        intensity = np.array([intensity1_vec[i], intensity2_vec[i]])

        train_set[i, :, :, :] = make_polygon_sequence(
            n,
            numEdge,
            line,
            r,
            rot,
            xoff,
            yoff,
            intensity,
            sequence_length,
            distort,
            outline,
            noise,
            rmod,
        )

    return train_set


def generator(
    num_ex=32,
    n=64,
    sequence_length=20,
    display=0,
    outline=False,
    noise=0,
    rmod=[1, 1],
    rot=[1 / 8, 1 / 8],
    num_edge=[4, 5],
    distort=0,
    r=[2, 8],
):
    # Creates a training set of sequences

    noise_vec = np.random.uniform(
        0, noise, num_ex
    )  # adds varying levels of noise to sequences
    numEdge_vec = np.random.randint(
        num_edge[0], num_edge[1], num_ex
    )  # number of edges in polygon
    distort_vec = np.random.uniform(distort, distort, num_ex)  # how much distortion
    outline_vec = np.random.uniform(0, 0.8, num_ex)  # 0 = filled in,
    r1_vec, r2_vec = np.random.uniform(r[0], r[1], num_ex), np.random.uniform(
        r[0], r[1], num_ex
    )
    rot1_vec, rot2_vec = np.random.uniform(rot[0], rot[1], num_ex), np.random.uniform(
        rot[0], rot[1], num_ex
    )
    xoff1_vec, xoff2_vec = np.random.uniform(
        n / 4, 3 * n / 4, num_ex
    ), np.random.uniform(n / 4, 3 * n / 4, num_ex)
    yoff1_vec, yoff2_vec = np.random.uniform(
        n / 4, 3 * n / 4, num_ex
    ), np.random.uniform(n / 4, 3 * n / 4, num_ex)
    intensity1_vec, intensity2_vec = np.random.uniform(1, 1, num_ex), np.random.uniform(
        1, 1, num_ex
    )

    train = make_polygon_batch(
        num_ex,
        n,
        sequence_length,
        numEdge_vec,
        distort_vec,
        outline_vec,
        r1_vec,
        r2_vec,
        rot1_vec,
        rot2_vec,
        xoff1_vec,
        xoff2_vec,
        yoff1_vec,
        yoff2_vec,
        intensity1_vec,
        intensity2_vec,
        outline,
        noise_vec,
        rmod,
    )

    return train


def create_test_set(r, name):
    train = generator(num_ex=500, sequence_length=20, r=r)
    test_path = os.getcwd() + name
    np.save(test_path, train)


if __name__ == "__main__":
    create_test_set([8, 21], "/square_test_set_20")
    create_test_set([8, 21], "/square_test_set_20_2")
    create_test_set([2, 8], "/square_test_set_20_train")
