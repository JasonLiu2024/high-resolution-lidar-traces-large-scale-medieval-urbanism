"""Display utility"""
import numpy as np
class Colors:
    def __init__(self) -> None:
        self.PURPLE = np.array([1.0, 0.1, 1.0]) # purple
        self.TURQUOISE = np.array([0.1, 1.0, 0.1]) # turquoise
        self.BLUE = np.array([0.1, 0.1, 1.0]) # blue
        self.GREEN = np.array([0.1, 1.0, 0.1]) # green
        self.YELLOW = np.array([1.0, 1.0, 0.1]) # yellow
        self.RED = np.array([1.0, 0.1, 0.1]) # red
        self.BLACK = np.array([0, 0, 0]) # black
        self.GREY = np.array([0.8, 0.8, 0.8]) # grey

def mesh_lines(vertices, faces, plot, color):
    plot.add_lines(vertices[faces[:, 0]], 
                    vertices[faces[:, 1]], 
                    shading={"line_color": color})
    plot.add_lines(vertices[faces[:, 0]], 
                    vertices[faces[:, 2]], 
                    shading={"line_color": color})
    plot.add_lines(vertices[faces[:, 1]], 
                    vertices[faces[:, 2]], 
                    shading={"line_color": color})
    
"""Plotter"""
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as mc
from matplotlib.transforms import Bbox
class Plot_2D():
    """ FOR: plot stuff, given plot–format {f, ax}, 
        init_color: (text) of 3D crestlines (projected to 2D)
        warp_color: (text) of 3D crestlines remapped by SEA"""
    def __init__(self, init_color, warp_color, thickness):
        self.init_color = init_color
        self.warp_color = warp_color
        self.thickness = thickness
    def Plot_ImageTransform(self, figure, axes, img_display):
        y_length, x_length = img_display.shape
        imgplot = axes.imshow(img_display, cmap=plt.cm.gray)
        transform = mpl.transforms.Affine2D().translate(-x_length/2, -y_length/2)
        transform = transform.scale(0.855) # need conversion
        transform = transform.rotate(np.deg2rad(-17.7447))
        transform = transform.translate(1470, 1260) # same amount as in blender
        imgplot.set_transform(transform + axes.transData)
    def Plot_ImageTransform_RGB(self, figure, axes, img_display):
        y_length, x_length, _ = img_display.shape
        imgplot = axes.imshow(img_display, cmap=plt.cm.gray)
        transform = mpl.transforms.Affine2D().translate(-x_length/2, -y_length/2)
        transform = transform.scale(0.855) # need conversion
        transform = transform.rotate(np.deg2rad(-17.7447))
        transform = transform.translate(1470, 1260) # same amount as in blender
        imgplot.set_transform(transform + axes.transData)
    def Plot_Handle(self, figure, axes, handle_dictionary, V):
        """ handle_dictionary: format: {vertex ID, target coordinates}
            V: list of vertex coordinates"""
        handle_coords_x = []
        handle_coords_y = []
        origin_coords_x = []
        origin_coords_y = []
        handle_number = 0
        for key in handle_dictionary.keys():
            x, y, _ = (handle_dictionary[key])
            handle_coords_x.append(x * 1000)
            handle_coords_y.append(y * 1000)
            x, y, _ = V[key]
            origin_coords_x.append(x * 1000)
            origin_coords_y.append(y * 1000)
            handle_number += 1
        axes.plot(origin_coords_x, origin_coords_y, 'o', markersize=12, color=self.init_color)
        axes.plot(handle_coords_x, handle_coords_y, 'o', markersize=12, color=self.warp_color)
    def Plot_Crestlines_Init(self, figure, axes, crestline_vertices, edges, conversion=1000):
        """ crestline_vertices: from 3D crestline
            warped_vertices: crestline_vertices after SEA remapping to 2D
            edges: crestline edges, for (u, v) format: [u index, v index, triangle ID]
            conversion: multiplier for vertex coordinates; here, I use 1000"""
        init_lines = [] # item = [(start coord tuple), (end coord tuple)]
        for i, j, _ in edges:
            x1, y1, _ = crestline_vertices[i] * conversion
            x2, y2, _ = crestline_vertices[j] * conversion
            init_lines.append([(x1, y1), (x2, y2)])
        init_line_collection = mc.LineCollection(init_lines, colors=self.init_color, linewidths=self.thickness, linestyle='solid')
        axes.add_collection(init_line_collection)
        axes.plot(crestline_vertices[:,0] * conversion, crestline_vertices[:,1] * conversion, 
                'o', markersize=self.thickness, color=self.init_color)
    def Plot_Crestlines_Warp(self, figure, axes, begin, end, conversion=1000):
        """ crestline_vertices: from 3D crestline
            warped_vertices: crestline_vertices after SEA remapping to 2D
            edges: crestline edges, for (u, v) format: [u index, v index, triangle ID]
            conversion: multiplier for vertex coordinates; here, I use 1000"""
        warp_lines = [] # item = [(start coord tuple), (end coord tuple)]
        for _, (s, e) in enumerate(zip(begin, end)):
            s1 = s[0] * conversion
            s2 = s[1] * conversion
            e1 = e[0] * conversion
            e2 = e[1] * conversion
            warp_lines.append([(s1, s2), (e1, e2)])
            # print(f"appending: {[(s1, e1), (s2, e2)]}")
        # for i, j, _ in edges:
        #     x1, y1, _ = warped_vertices[i] * conversion
        #     x2, y2, _ = warped_vertices[j] * conversion
        #     warp_lines.append([(x1, y1), (x2, y2)])

        warp_line_collection = mc.LineCollection(warp_lines, colors=self.warp_color, linewidths=self.thickness, linestyle='solid')
        axes.add_collection(warp_line_collection)
        # axes.plot(warped_vertices[:,0] * conversion, warped_vertices[:,1] * conversion, 
        #         'o', markersize=self.thickness, color=self.warp_color)
    def Plot_Labels(self, figure, axes):
        COURIER = {'fontname':'Courier'}
        axes.text(0.01, 0.95, 'Original crestlines',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes.transAxes,
            color=self.init_color, fontsize=24, **COURIER)
        axes.text(0.01, 0.90, 'WARPed crestlines',
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes.transAxes,
            color=self.warp_color, fontsize=24, **COURIER)
        axes.text(0.99, 0.95, 'Point = handle  ',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes.transAxes,
            color='white', fontsize=24, **COURIER)
        axes.text(0.99, 0.90, 'Line = crestline',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes.transAxes,
            color='white', fontsize=24, **COURIER)
    def Plot_SaveTrim(self, figure, axes, snapshot, filename, lower_left, lower_right, dpi=169.18):
        """ DOES: save ONLY what's INSIDE axes
            snapshot: do a plt.gcf BEFORE draw() and show()
            lower_left: lower left corner of bounding box
            lower_right: lower right corner"""
        # source: https://stackoverflow.com/questions/64676770/save-specific-part-of-matplotlib-figure
        x0, y0 = lower_left
        x1, y1 = lower_right
        bbox = Bbox([[x0, y0],[x1, y1]])
        bbox = bbox.transformed(axes.transData).transformed(figure.dpi_scale_trans.inverted())
        snapshot.savefig(filename, dpi=dpi, bbox_inches=bbox, pad_inches=0, facecolor=figure.get_facecolor())
    def Plot_SaveAxes(self, figure, axes, snapshot, filename, dpi=169.18):
        """ DOES: save figure with axes
            snapshot: do a plt.gcf BEFORE draw() and show()"""
        snapshot.savefig(filename, dpi=dpi, bbox_inches='tight')
    def Plot_Background(self, figure, axes, img_display, background_color):
        """ DOES: fills background with color
            background_color: (All-Caps text)"""
        image = Image.fromarray(img_display)
        blank = Image.new(mode="RGBA", size=image.size, color=background_color)
        axes.imshow(blank, cmap=plt.cm.gray)
    # def Compare(self, figure, axes, image, size, background, handle, crestline, label):
    #     """ DOES: compare images against other items
    #         image: image being shown
    #         size: (int, int) how big ea image
    #         other items: (bool) whether this item shows up (in plot)"""
    #     # figure, axes = plt.subplots(figsize=(32, 32)) # NOT subplot()
    #     # print(f"image is {x_length}, {y_length}")
    #     # axes.set_xlim(0, x_length)
    #     # axes.set_ylim(0, y_length)
    #     if(background == True): self.Plot_Background(figure, axes, "BLACK")
    #     self.Plot_ImageTransform(figure, axes, image)
    #     if(handle == True): self.Plot_Handle(figure, axes, handle_dictionary, V)
    #     if(crestline == True): self.Plot_Crestlines(figure, axes, crestline_V_3d, 
    #                 mover.mesh_2d_crestlines_vertices, crestline_E)
    #     if(label == True): self.Plot_Labels(figure, axes)

