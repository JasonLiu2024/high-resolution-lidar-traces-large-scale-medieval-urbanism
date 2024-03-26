Fast and Robust Detection of Crest Lineson Meshes C++ code & Java3D Viewer

 Shin Yoshizawa (shin@riken.jp)

************************
This C++ codes & Java3D Viewer are developed by Shin Yoshizawa at the MPII, Saarbruecken, Germany. The method is described in my paper "Fast and Robust Detection of Crest Lineson Meshes", Shin Yoshizawa, Alexander G. Belyaev, and Hans-Peter Seidel.

  Since April 2002, Shin Yoshizawa had been working
at Max-Planck-Institut fuer Informatik: Computer Graphics Group, 

 Since January 2007, Shin Yoshizawa moved to RIKEN: VCAD, Bio-research Infrastructure Construction Team Hirosawa 2-1, Wako, Saitama, Japan. 

 The C++ code, Java3D Viewer, and their simple manual are available at 
http://www.riken.jp/brict/Yoshizawa/Research/Crest.html



************************
Copyright
************************
Copyright:(c) Shin Yoshizawa, 2011.


 All right is reserved by Shin Yoshizawa.
This C++ code files and Java3D Viewer are allowed 
for only primary user of research and educational purposes. Don't use 
secondary: copy, distribution, diversion, business purpose, and etc.. 

 In no event shall the author be liable to any party for direct, indirect, 
special, incidental, or consequential damage arising out of the use of this 
program and source files. 

(*): The functions of the class SvdSolve and Eigens
       are written in "Numerical Recipes in C++", The Art of Scientific Computing, William H. Press , Saul A. Teukolsky, William T. Vetterling, Brian
       P. Flannery. The copyrights of these functions of SvdSolve and Eigens class are remained to them. 
************************
Commercial and Business Use
If you would like to use this C++ and Java sources in your commercial software then please contact with me and RIKEN.

shin@riken.jp
************************

************************
Files in CrestCODE
************************
Crest.html: simple manual (copy of http://www.mpi-sb.mpg.de/~shin/Research/Crest/Crest.html) images and links are not appropriate.
CrestViewer.jar: Java3D viewer for UNIX and linux users. 
CrestViewerWindows.jar: Java3D viewer for Windows users. 
moai.ply2: sample triangle mesh scanned by Dr. Ohtake. 
CCode:Makefile and C++ source files.
Edge.h      IDSet.cxx  Point3d.h      Polyhedron.cxx  setCurvature.cxx
Eigens.cxx  IDSet.h    PointTool.cxx  Polyhedron.h
Eigens.h    Makefile   PointTool.h    SvdSolve.cxx
IDList.h    NRmacro.h  PolarList.h    SvdSolve.h
************************

(*) the compiled C++ executable file should be same directory of Java viewer (Jar).
************************
The program is using "stdio.h" and "math.h".
You can comple "make all" via Makefile.
************************
Simple Manual

      How to use
          o Program Organization: The program is constructed by two parts: C++ and Java3D (CrestViewer.jar). The C++ program do actual computations to detect crest lines: normal estimation, polynomial fitting, curvature derivatives calculation, crest line tracing, and Thresholds computations (Ridgeness, Sphericalness and Cyclideness). The Java3D program can visualize the detected crest lines, and can produce an input file for the C++ code easily. Of cause you can use only the C++ code but you have to write the input and visualization parts by yourself. CrestCODE.tar.gz includes a directory: CCode. The CCode directory has all C++ files and Makefile.
          o Compile:
                + C++ code, by using attached Makefile, you may run "make" in CCode directory. Then you will get the executable file "setCurvature". Please move setCurvature to the JViewer directory (% mv setCurvature ../JViewer/) if you would like to use the Java3D Viewer. Otherwise you can deal with the setCurvature file. You should use the later versions of g++ 2.95.
          o Execute:
                + C++ code, You can run setCurvature itself as (% ./setCurvature input.txt output.txt). Then you will get also ridges.txt and ravines.txt which include the information about convex and concave crest lines. The output.txt includes the curvature tensor information. The formats of input.txt, output.txt, ridges.txt and ravines.txt are described in File Formats section of this simple manual.
                + Java3D Viewer,
                  You need JDK and Java3D API environment. You may run java (% java CrestViewer.jar). It is better to specify max and initial heap sizes as (% java -mx600m -ms600m -jar CrestViewer.jar) to avoid OutofMemoryError. Please read the Quick Practice and Viewer Options sections to play the Java3D Viewer. Note that the executable file setCurvature should be located in same directory of CrestViewer Jar file. File Formats:
          o
                + Input for Java3D Viewer: The input triangle mesh have to be constructed by the PLY2 format.
                + Recall, when we apply "./setCurvature input.txt output.txt" which also generates ridges.txt and ravines.txt. These .txt files are ascii text.
                      # Input of C++ code (input.txt): Similar format of the PLY2 format but it is not an exact PLY2 format. 1st line is a number of vertices. 2nd line is a number of triangles. 3rd line is a neighborhood size (Integer) for fitting. 4th line is 0 (no crest line tracing) or 1 (with crest line tracing). After these four lines, there are vertex coordinates and face information as PLY2 format but there is no 3 for face line (only three vertex IDs).
                      # Outputs of C++ code:
                            * output.txt: 1st line is a number of vertices (Integer). 2nd line is a number of triangles (Integer). Starting from 3nd line, 11 Double values in one line: k_{max}, k_{min}, x,y,z of t_{max}, x,y,z of t_{min}, and x,y,z of normal for each vertex ID of input mesh.
                            * ridges.txt/ravines.txt: 1st line is a number of convex/concave crest line points (Integer). 2nd line is a number of convex/concave crest line edges. 3rd line is a number of crest line connected components (Integer). Let us notate V, E, and N for above numbers, respectively. Starting from 4th line, there are V lines which include three Double and one Integer values in one line as "%lf %lf %lf %d": x,y,z of crest line point and the connected component ID. The line number of the file minus 4 represents a crest point ID. Next, there are N lines which include three Double values in one line: Ridgeness, Sphericalness and Cyclideness for each corresponded connected component ID (ex. 1st line of this section corresponds ID 0 of the connected component, 2nd line of them represents ID 1 and so on). Finally, there are E lines which include three Integer values: pair of crest points ID (edge) and the triangle ID of original mesh which includes this edge (if there is -1 of that triangle ID then that edge is a connecting edge, see the paper).
          o For Windows User: Please use CrestViewerWindows.jar instead of CrestViewer.jar because you may have setCurvature.exe instead of setCurvature. If you have setCurvature then it is OK to use CrestViewer.jar on the Windows OS. Cygwin may help to compile my code on Windows. 
      Quick Practice
      Note that the C++ executable file (setCurvature) and the Java Jar file (CrestViewer.jar) should be in the same directory.
          o Road to ready:
            % gunzip CrestCODE.tar.gz
            % tar -xf CrestCODE.tar
            % cd ./CrestCODE/CCode
            % make
            % mv setCurvature ../
            % java -mx600m -ms600m -jar CrestViewer.jar
          o Let's play the crest lines Java3D Viewer:
            Click the top menu: file->load->load (poly2)
            Select the file moai.ply2.
            Click "Compute Crests" button.
            Click "Material Light" checkbox.
            Scroll "Cyclideness" scrollbar.
      Viewer Options
          o Menubar:
                + file->exit program: to exit program.
                + file->load:
                      # load (poly2): load an input PLY2 formated triangle mesh.
                      # load Ridge/load Ravine: load a set of convex/concave crest lines. Note that you have to load an appropriate original mesh before load crest lines.
                      # load view: load a view point.
                + file->save:
                      # save (poly2): save the current triangle mesh by PLY2 format.
                      # save Ridge/load Ravine: save a set of convex/concave crest lines. Note that a all set of crest lines would be saved (they are not current visible crest lines). The file format is save of ridges.txt/ravines.txt which are described in above section of this simple manual.
                      # save view: save a view point.
                      # save image (.jpg): capture a screen image in the canvas region via JPEG.
                + Original Mesh->exist: On: visualize the original mesh. Off: unbind the original mesh.
                + Original Mesh->Normal Color: Coloring the mesh via the original one color.
                + Original Mesh->Curvature-Pastel: Coloring the mesh via curvatures: mean (M), Gaussian (K), maximum principal (max) and minimum principal (min) curvatures according to the selected checkbox in the Visualization Type Panel. Note that you have to apply "Compute Crests" button at least one time before selecting this checkbox menu. You can dynamically visualize the curvature via selecting these checkbox in the Visualization Type Panel after select this checkbox menu. To go back normal coloring or white coloring, you may click "Material Light" checkbox twice in the Visualization Type Panel. It is also useful to scroll the "Curvature s-deviation" scrollbar which is thresholding to visualize appropriate curvature value.
                + Original Mesh->WireFrame: visualize the mesh via wire frame model. To go back the face rendering, you may select "Original Mesh->Normal Color" checkbox menu.
                + Original Mesh->Points: visualize the mesh vertices. To go back the face rendering, you may select "Original Mesh->Normal Color" checkbox menu.
                + Original Mesh->2 Light: extra lighting on/off.
                + Original Mesh->Back Face: visualize the back face of the mesh on/off.
                + Original Mesh->Transparency: half-transparent rendering of the mesh on/off.
                + Camera: select camera positions. Nate that it also changes orientation for mouse motions.
          o Visualization Type Panel:
                + Material Light: material light and shrinking the mesh on/off. To visualize the crest lines more perceptually, this checkbox affects the mesh geometry, lighting and coloring.
                + M, K, max, and min: visualization curvature types when you select "Original Mesh->Curvature-Pastel" checkbox menu.
                + N, tmax, tmin, R and V: visualize (on/off) the normal (N), maximum principal (tmax) and minimum principal (tmin) directions, convex crest line (R), and concave crest line (V). Note that you have to apply "Compute Crests" button at least one time before selecting this checkbox. Also, if you would like to compute only curvatures and directions then you can select off of "R" and "V" before click the "Compute Crests" button. However you have to apply "Compute Crests" button again (re-calculation) to visualize crest lines.
                + Compute Crests: Calculate curvature tensor, crest lines and thresholding values.
                + N-rings N: Neighborhood size for the polynomial fitting (Integer > 0). N-ring neighborhood is applied for the fitting.
          o Mesh Processing Tool Panel: This panel is not directly related to the crest lines detection and filtering excepting "FRT" checkbox.
                + Smoothing: apply smoothing to the mesh. St = T,LL: bitangential Laplacian. St = L: Laplacian. St = L(L): bilaplacian. I: initialize vertex positions. Note that the smoothing result does not affect to the crest line detection if the checkbox "C" is off. If the checkbox "C" is on then the smoothing result will send to processing of the crest line detection when you click "Compute Crests". TextField represents a number of smoothing iterations.
                + Subdivide: apply subdivision to the mesh via Loop, Butterfly, or Linear. The subdivision results overwrite the original mesh. Loop N: Loop subdivision without projection to the limit position. Loop: Loop with limit position projection. TextField represents a number of subdivision iterations. Note that please do not apply "Subdivide" the large mesh or many times iterations because of my memory expensive implementation (you will have OutofMemoryError after a long time waiting).
                + GH-Decimation: apply Garlnd-Heckbert mesh simplification. "P:" is a reduction ratio (percentage). The decimation result overwrites the original mesh.
                + FRT: This checkbox affects the crest line thresholding when you scroll the scrollbars. On: the scrollbar dynamically affects the visualization result. Off: the scrollbar does not affects the visualization result. It might be useful for the huge meshes because you can change a value of thresholding without visualization, then you can update that value by clicking the triangle buttons of the left and right sides of the scrollbar after activate (off -> on) this checkbox.
          o ScrollBars (Filtering Crest Lines): you have to apply "Compute Crests" button at least one time before changing the following scrollbars.
                + Curvature s-deviation: thresholding value for the curvature visualization, see "Original Mesh->Curvature-Pastel" section. It is better to change this value when the mesh is colored almost one color for curvature visualization.
                + Ridgeness: changing ridgeness which is integration of the curvature along the connected crest lines.
                + Sphericalness: changing sphericalness which is average of the window functioned Shapeindex for the connected crest lines (0 < s <1).
                + Cyclideness: changing cyclideness which is integration of the MVS (minimum variation surface) functional along the connected crest lines, see the paper).


