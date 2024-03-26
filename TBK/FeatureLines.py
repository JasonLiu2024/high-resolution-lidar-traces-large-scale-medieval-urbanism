"""Functions: for raw .ply input!"""
import open3d as o3d
import numpy as np
# input: mesh file (path)
# output1: vertices (N, 3) array; x y z coordinates
# output2: faces (F, 3) array, format: [a id, b id, c id]
def get_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    print(mesh)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces
# input: output file name, neighbor (k), crestline (1 = yes, 0 = no), vertices (array), faces (array)
# does: txt file <- for CrestCODE
def to_pseudo_PLY2(name, neighbor, verts, faces, crestline=1):
    NEWLINE = '\n'
    verts_count = len(verts)
    faces_count = len(faces)
    with open(name, 'w') as f:
        f.write(str(verts_count) + NEWLINE)
        f.write(str(faces_count) + NEWLINE)
        # print(f"wrote neighbor value: k = {neighbor}")
        f.write(str(neighbor) + NEWLINE)
        f.write(str(crestline) + NEWLINE)
        for i in range(verts_count): # all vertices x y z coordinates
            f.write(" ".join(map(str, verts[i])) + NEWLINE)
        for j in range(faces_count): # all faces a b c vertices ids
            f.write(" ".join(map(str, faces[j])) + NEWLINE)
    print(f"Success: {name}")
"""Functions: for CrestCODE output"""
# get data from Ridges.txt OR Ravines.txt
# input: meshfile name, mesh vertices, mesh faces
# output1: crest line vertices (V, 4) array, format: [x coordinate, y coordinate, z coordinate, connected component]
# output2: connected components (N, 3) array, format: [Ridgeness, Sphericalness, Cyclideness]
# output3: crest line edges (E, 3) array, format: [u vertex index, v vertex index, triangle ID]
def ReadCrestLine(filename):
    f = open(filename, 'r')
    V = int(f.readline())
    E = int(f.readline())
    N = int(f.readline()) # num of connected components;
    crestline_vertices = np.zeros(shape = (V, 4)) # crest line vertices
    for i in range(V): 
        line = f.readline()
        crestline_vertices[i] = [float(n) for n in line.split()]
    crestline_connected_cmp = np.zeros(shape = (N, 3)) # index = connected cmp ID
    for j in range(N):
        line = f.readline()
        crestline_connected_cmp[j] = [float(n) for n in line.split()]
    # edges (u,v): [vtx ID of u, vtx ID of v, triangle ID]
    crestline_edges = np.zeros(shape = (E, 3), dtype=int) # index = edge ID
    for k in range(E):
        line = f.readline()
        crestline_edges[k] = [n for n in line.split()]
    return crestline_vertices[:,0:3], crestline_edges