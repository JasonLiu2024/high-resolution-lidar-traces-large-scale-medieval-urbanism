# Frachetti et al. (Under review: Nature_2024-03-05358)
Code for "*Automated analysis of high-resolution lidar traces large-scale medieval urbanism in highland Central Asia*" by Michael Frachetti, Jack Berner, Jason Liu, Edward Henry, Tao Ju, Farhod Maksudov (Pre-print in review: https://doi.org/10.21203/rs.3.rs-4108554/v1)

## 1. System requirements
### Software dependencies
- ```requirements.txt``` Lists required python packages
- ```CrestCODE/```(C++ project, included and compiled on our machine) detects crest lines from a mesh. (source: http://www2.riken.jp/brict/Yoshizawa/Research/Crest.html)
- ```SEA/```(C++ project, included and compiled on our machine) deforms a mesh with handles. Handles are positional constraints indicating where a point (vertex) in the starting mesh should be in the deformed mesh. The deformed mesh does not self-intersect. (source: https://github.com/duxingyi-charles/Smooth-Excess-Area)
### Tested on: macOS Somona Version 14.4. 
Our machine is an Apple M1 Pro chip with 16GB RAM.
### No non-standard hardware required
> [!NOTE]
> Dataset: Run time of ```CrestCODE``` and ```SEA``` are impacted by by the size of the mesh.

## 2. Installation guide
### Instructions (with install time on our machine)
1. (~15 minutes) Create a Conda environment with Python version of 3.9.16
2. (~5 minutes) Compile ```CrestCODE``` and ```SEA``` in their respective folders.

## 3. Demo
### Preprocessing
- ```point_cloud_to_mesh.ipynb``` Generates mesh from a point cloud. Prior georeferencing of the point cloud carries over to this mesh.
<br><b>Expected output</b>: mesh with triangles, such as:
<br>![](/pictures/tgb_mesh_raw.png)
<br>Our code optionally removes parts of the mesh with low point density.

> [!NOTE]
> Dataset: we provide ```TBK/TBK_mesh.ply``` and ```TGB/TGB_mesh.ply```, which are produced through this file.

### Tashbulak (TBK) results reproduction
(Timed based on provided data)
1. (~10 minutes)```1_TBK_crest_lines.ipynb```
(1) Generates crest lines from a mesh with ```CrestCODE/``` and (2) Deforms the mesh using ```SEA/```
<br><b>Expected output</b> (these will be rendered as you run the file)
: 
<br>(1) crest lines, such as (red crest lines on white shaded mesh):
<br>![](/pictures/tbk_mesh_with_crest_lines.png)
<br>(2) deformed mesh that is completely flat. Crest lines (that lie within triangles on the mesh) are deformed alongwith and rasterized. The bitmap (rendered as an image) looks like (white against black background):
<br>![](/pictures/tbk_crest-lines_deformed.png)

> [!IMPORTANT]
> Centerline extraction is affected by the size of the GPR image.

2. (~4 hours)```2_TBK_GPR.ipynb``` extracts centerline (in bitmap) from GPR images
<br><b>Expected output</b>: 
<br>(1) scaled and rotated GPR images in black and white, such as:
<br>![](/pictures/tbk_radar.png)
<br>Bitmap of the composite of centerlines from the input GPR images, such as (red against the above GPR image):
<br>![](/pictures/tbk_centerlines_on_radar.png)

3. (~10 minutes)```3_TBK_validation.ipynb``` computes similarity of <b>crest lines</b> to <b>centerlines</b>, with statistics and illustration.
<br><b>Expected output</b>: statistics ('hit' rate, confusion matrix, ect.) with illustrations like (the colors can be adjusted): 
<br>![](/pictures/tbk_confusion_matrix.png)

4. (~5 minutes)```4_TBK_crestlines_to_shapefile.ipynb``` converts crest lines to Shapefile format.
<br><b>Expected output</b>: crest lines in Shapefile (the projection could be adjusted, and images previews could be exported)

### Tugunbulak (TBK) results reproduction
(Timed based on provided data)
#### Code to run:
1. (~10 minutes)```1_TGB_crest_lines.ipynb``` Generates crest lines from a mesh with ```CrestCODE/```
<br><b>Expected output</b>: image like
<br>![](/pictures/tgb_mesh_crest_lines.png)

> [!TIP]
> To save storage space of georeferenced point clouds, we can center the coordinates of all the points. This retains the positions of the points relative to each other but with much smaller coodrinate values. That results less than 1/2 the file size of the original point cloud. Of course, the translation due to the centering should be recorded to restore the results, such as crest lines, that are much more light-weight.

2. (~5 minutes)```2_prepare_human_vs_crest_lines.ipynb``` Rasterize human tracings and crest lines.
<br><b>Expected output</b>: bitmap of crest lines human tracings, such as:
<br>Crest lines bitmap
<br>![](/pictures/tgb_crestlines_bitmap.png)
<br>Human tracings bitmap
<br>![](/pictures/tgb_traced-lines_bitmap.png)

3. (~10 minutes)```3_TGB_validation.ipynb``` computes similarity of <b>crest lines</b> to <b>human tracings</b> (in bitmap form) Berner, with statistics and illustration.
<br><b>Expected output</b>: statistics ('hit' rate, confusion matrix, ect.) with illustrations like (from the above tracings and crest lines): 
<br>![](/pictures/tgb_confusion_matrix.png)

4. (~5 minutes)```4_TGB_crestlines_to_shapefile.ipynb``` converts crest lines to Shapefile format.
<br><b>Expected output</b>: crest lines in Shapefile (the projection could be adjusted, and images previews could be exported)
