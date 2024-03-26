"""deform"""
import numpy as np
import open3d as o3d
def Handles(mesh_vertices, mesh_faces, affected_vertex_ids, target_positions, iterations):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_vertices),
                                 o3d.utility.Vector3iVector(mesh_faces))
    # constraint_ids = o3d.utility.IntVector(affected_vertex_ids)
    # constraint_pos = o3d.utility.Vector3dVector(target_positions)
    deform = mesh.deform_as_rigid_as_possible(
        constraint_vertex_indices=o3d.utility.IntVector(affected_vertex_ids), 
        constraint_vertex_positions=o3d.utility.Vector3dVector(target_positions), 
        max_iter=iterations)
    return np.asarray(deform.vertices)