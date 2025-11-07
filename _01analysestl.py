from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np
import sys

# Defaults
file_stl = "test.stl"
file_cuts = "cuts.txt"

def analyseSTL(in_stl, out_cuts, cutoffangle = 45, mincutheight = 5, debug = False):
    """
    Decide the cutting heights for a given stl file
    Faces steeper than the cuttoffangle are grouped together in z-direction if continuous. 
    :param in_stl: string
        path to the stl file
    :param out_cut: string
        path of a txt files where the different cutting heights will be stored separated by newlines
    :param cutoffangle: degree
        Faces steeper than cutoffangle are grouped together and marked to be cut out
    :param mincutheight: mm
        Minimal height of resulting heights of slices
    :param debug: bool
        Show debug information
    :return: 
    """
    # Get face angles
    body = mesh.Mesh.from_file(in_stl)
    normals = body.get_unit_normals()
    normals_z = normals[:,2]
    angle = np.arccos(normals_z)

    # Get all faces steeper than cutoffangle and sort them in z-direction
    zmin = [min(face[:,2]) for face in body.vectors]
    zmax = [max(face[:,2]) for face in body.vectors]
    cutoffangle = cutoffangle / 180 * np.pi + np.pi/2
    interval = [[min, max, a] for (min, max, a) in zip(zmin, zmax, angle) if a > cutoffangle]
    interval.sort(key=lambda x: x[0])
    if(debug): print("Sorted intervals of faces steeper than ", cutoffangle, "°:\n", interval)

    # Group continuous intervals together
    index = 0
    for i in range(1, len(interval)):
        if (interval[index][1] >= interval[i][0]):
            interval[index][1] = max(interval[index][1], interval[i][1])
        else:
            index = index + 1
            interval[index] = interval[i]
    interval = interval[0:index+1]
    if(debug): print("Continuous intervals:\n", interval)

    # Calculate cutheights according to intervals and mincutheight
    minheight = min(zmin)
    maxheight = max(zmax)
    cutheights = []
    cutheights.append(minheight)
    if interval[0][1]-interval[0][0] > 0.2:
        cutheights.append(max(interval[0][1], interval[0][0]+mincutheight))
    interval.append([maxheight, maxheight])
    for i in range(1, len(interval)-1):
        if(cutheights[-1] >= interval[i][0]):
            cutheights[-1] = interval[i][0]
        else:
            cutheights.append(interval[i][0])
        cutheights.append(max(interval[i][1], interval[i][0] + mincutheight))
    if(cutheights[-1] > maxheight):
        del cutheights[-1]
    if(debug): print("cutheights:\n", cutheights)

    # Save cuts in file
    with open(out_cuts, "w") as f_cuts:
        for cut in cutheights:
            f_cuts.write(str(cut) + "\n")#" ".join(cut) + "\n")


    if(debug):
        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        col = mplot3d.art3d.Poly3DCollection(body.vectors)
        color = [(1,0,0) if a > cutoffangle else (0,0,1) for a in angle]
        col.set_facecolor(color)
        axes.add_collection3d(col)

        # Auto scale to the mesh size
        scale = body.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        pyplot.show()

if __name__ == "__main__":
    if(len(sys.argv) > 2):
        file_stl = sys.argv[1]
        file_cuts = sys.argv[2]
    if(len(sys.argv) > 5):
        analyseSTL(in_stl, out_cuts, sys.argv[3], sys.argv[4], True)
    elif(len(sys.argv) > 4):
        analyseSTL(in_stl, out_cuts, sys.argv[3], sys.argv[4])
    else:
        analyseSTL(in_stl, out_cuts)