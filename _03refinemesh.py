import numpy as np

from stl import mesh

# Defaults
file_stl = "test.stl"
edgelength = 3

def refineAllTriangles(triangles, maxlength, singlerun = False):
    '''
    Splits the triangle so the two longer edges are smaller than maxlength
    :param triangles: array
        array of triangles
    :param maxlength: float
        Maximum length of an edge
    :param singlerun: bool
        If the algorithm should only be run once. When running it twice all edges are smaller than maxlength
    :return array
        refined triangles
    '''
    #print("Original triangles:\n", triangles)
    
    # get edge length
    triangles2D = triangles[:,:,0:2]
    triangles2D = triangles2D.reshape((-1,3,2))
    length1 = np.sqrt(np.sum(np.subtract(triangles2D[:,1,:], triangles2D[:,0,:])**2, 1))
    length2 = np.sqrt(np.sum(np.subtract(triangles2D[:,2,:], triangles2D[:,1,:])**2, 1))
    length3 = np.sqrt(np.sum(np.subtract(triangles2D[:,0,:], triangles2D[:,2,:])**2, 1))
    #print("Edge 1 length:\n", length1)
    #print("Edge 2 length:\n", length2)
    #print("Edge 3 length:\n", length3)
    length = np.reshape(np.column_stack([length1, length2, length3]), (-1, 3))
    #print("All edges length:\n", length)
    
    # shortest edge
    shortest = np.argmin(length, 1)
    #print("Shortest edges:\n", shortest)

    # get number of points per edge
    numtriangles = shortest.size
    numvertex = np.empty((numtriangles, 3), int)
    np.ceil(np.divide(length, maxlength), out=numvertex, casting='unsafe')
    #print("Number of vertices per edge:\n", numvertex)

    # adjust number of vertices on shortest edge
    sortednumvertex = np.sort(numvertex, 1)
    numvertexshortedge = np.maximum(1, sortednumvertex[:,2]-sortednumvertex[:,1])
    #print("Number of vertices on short edge:\n", numvertexshortedge)

    # get new edge order
    order = np.empty(3, int)
    refinedtriangles = []
    for i in range(numtriangles):
        if (shortest[i] == 0):
            order[0] = 2
        else:
            order[0] = shortest[i]-1
        order[1] = shortest[i]
        if (shortest[i] == 2):
            order[2] = 0
        else :
            order[2] = shortest[i]+1
        #print("Order of vertices:\n", order)

        # get all the points of the new triangles
        vertices = np.concatenate([np.linspace(triangles[i,order[0],:], triangles[i,order[1],:], numvertex[i,order[0]], endpoint=False),
                    np.linspace(triangles[i,order[1],:], triangles[i,order[2],:], numvertexshortedge[i]),
                    np.linspace(triangles[i,order[2],:], triangles[i,order[0],:], numvertex[i,order[2]], endpoint=False)])
        #print("New vertices:\n", vertices)

        # triangulate
        index1 = 0
        index3 = np.size(vertices, 0) - 1
        while(index3-index1 > 1):
            refinedtriangles.append([vertices[index1], vertices[index1+1], vertices[index3]])
            index1 += 1
            if(index3-index1 > 1):
                refinedtriangles.append([vertices[index1], vertices[index3-1], vertices[index3]])
                index3 -= 1
    #print("Refined triangles:\n", np.array(refinedtriangles))
    
    refinedtriangles = np.array(refinedtriangles)
    if (not singlerun):
        refinedtriangles = refineAllTriangles(refinedtriangles, maxlength, True)

    return refinedtriangles





def refineMesh(in_file, maxlength=3):
    '''
    Refines an stl file so all edges are smaller than maxlength
    :param in_file: string
        STL-file containing the triangles. The results gets stored here as well
    :param maxlength: float
        Maximum length of an edge in mm
    :return
    '''
    body = mesh.Mesh.from_file(in_file).vectors
    triangles = body
    newtriangles = refineAllTriangles(triangles, maxlength)
    newmesh = np.zeros(newtriangles.shape[0], dtype=mesh.Mesh.dtype)
    newmesh['vectors'] = newtriangles
    newmesh = mesh.Mesh(newmesh)
    newmesh.save(in_file)

if __name__ == "__main__":
    if(len(sys.argv) > 1):
        file_stl = sys.argv[1]
    if(len(sys.argv) > 2):
        edgelength = sys.argv[2]
    refineMesh(file_stl, edgelength)