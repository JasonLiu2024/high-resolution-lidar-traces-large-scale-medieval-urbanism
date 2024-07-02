"""NOTE: this script is run via Blender
website: https://www.blender.org
"""
import bpy

here = 'current directory'

landmarks_mesh_file = open(f'{here}/name of landmarks file.txt', 'w')

for o in bpy.data.collections.get('name of collection that contains the landmarks').all_objects:
    l = o.location
    landmarks_mesh_file.write(f'{o.name}, {l[0]},{l[1]},{l[2]}\n')