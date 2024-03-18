import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
from FeatureLines import ReadCrestLine
import matplotlib.pyplot as plt

def undo_centering(names_readable : list[str], list_of_lines : list[LineString], 
  movement_to_undo : list[float], name_to_filename_dictionary : dict[str, dict]):
  for name in names_readable:
    # print(f"working on: {name}")
    filename = f"{name_to_filename_dictionary[name]['filename']}.txt"
    # print(f"filename: {filename}")
    open(filename)
    crestline_V_3d, crestline_E = ReadCrestLine(f"{name_to_filename_dictionary[name]['filename']}.txt")
    # print(f"we got: \n\t{crestline_V_3d.shape[0]} points, \n\t{crestline_E.shape[0]} lines")
    if movement_to_undo != [0.0, 0.0, 0.0]:
        for (start_index, end_index, _) in crestline_E:
            start_point = [
                crestline_V_3d[start_index][0] + movement_to_undo[0],
                crestline_V_3d[start_index][1] + movement_to_undo[1]]
            end_point = [
                crestline_V_3d[end_index][0] + movement_to_undo[0],
                crestline_V_3d[end_index][1] + movement_to_undo[1]]
            line = LineString([start_point, end_point])
            list_of_lines.append(line)  

from operator import itemgetter
def compare_two(region_names_list : list[list[str]], 
    colors : list[str], display_names : list[str],
    centering_movement_to_undos : list[list[float]],
    name_to_filename_dictionary : dict[str, dict],
    bounding_region_index : int=None,
    bounds_to_use : tuple[float, float, float, float]=None,
    margin : int=2,
    style : str='one picture',
    figure_height : int = 10,
    figure_title : str='no title given',
    background_line_shapefile_name : str=None,
    save_name : str=None):
    assert bounding_region_index < len(region_names_list), "bounding region not in list of regions!"
    gpd.options.display_precision = 6
    all_lines : list[list[LineString]]= []
    all_frames : list[gpd.GeoDataFrame]= []
    for (i, region_name) in enumerate(region_names_list):
        lines : list[LineString] = []
        undo_centering(region_name, lines, centering_movement_to_undos[i], name_to_filename_dictionary)
        all_lines.append(lines)
        all_frames.append(gpd.GeoDataFrame(data={'geometry': lines}, crs="EPSG:4326"))
    if bounds_to_use: # bounds given
        minx, miny, maxx, maxy = bounds_to_use
    else: # calculate bounds from items to display
        if bounding_region_index == None:
            bounding_region_index=0
        bounds = all_frames[bounding_region_index].geometry.apply(lambda x: x.bounds).tolist()
        minx, miny, maxx, maxy = min(bounds, key=itemgetter(0))[0], min(bounds, key=itemgetter(1))[1], max(bounds, key=itemgetter(2))[2], max(bounds, key=itemgetter(3))[3]
    if style == 'side by side':
        f, ax = plt.subplots(figsize=(figure_height * len(region_names_list), figure_height), ncols=len(region_names_list))
        f.suptitle(figure_title, y=1.00)

        if len(region_names_list) == 1:
            ax.set_xlim(xmin=minx - margin, xmax=maxx + margin)
            ax.set_ylim(ymin=miny - margin, ymax=maxy + margin)
            print(f"x bounds: {minx - margin, maxx + margin}")
            print(f"y bounds: {miny - margin, maxy + margin}")
            all_frames[0].plot(aspect=1, color=colors[0], ax=ax)
            if background_line_shapefile_name:
                background_frame = gpd.read_file(background_line_shapefile_name)
                background_frame.plot(aspect=1, color='red', ax=ax)
            ax.set_title(display_names[0])
        else:
            for (j, region_name) in enumerate(region_names_list):
                ax[j].set_xlim(xmin=minx - margin, xmax=maxx + margin)
                ax[j].set_ylim(ymin=miny - margin, ymax=maxy + margin)
                print(f"x bounds: {minx - margin, maxx + margin}")
                print(f"y bounds: {miny - margin, maxy + margin}")
                # ax[j].set_xlim(xmin=minx, xmax=maxx)
                # ax[j].set_ylim(ymin=miny, ymax=maxy)
                all_frames[j].plot(aspect=1, color=colors[j], ax=ax[j])
                if background_line_shapefile_name:
                    background_frame = gpd.read_file(background_line_shapefile_name)
                    background_frame.plot(aspect=1, color='red', ax=ax[j])
                ax[j].set_title(display_names[j])
                # ax[j].set_axis_off()
        f.tight_layout()

        # f.subplots_adjust(top=1.4)
    else:
        f, ax = plt.subplots(figsize=(24, 16))
        ax.set_xlim(xmin=minx - margin, xmax=maxx + margin)
        ax.set_ylim(ymin=miny - margin, ymax=maxy + margin)
        for (i, frame) in enumerate(all_frames):
            frame.plot(aspect=1, color=colors[i], ax=ax, label=display_names[i])
        ax.legend()
    if save_name:
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        DPI=100
        current_figure.savefig(f"{save_name}", dpi=DPI, pad_inches=0, facecolor=f.get_facecolor(), ) # bbox_inches='tight'
        print(f"saved figure {save_name}, DPI = {DPI}")
    return minx, miny, maxx, maxy

from matplotlib.patches import Rectangle
def show_boundary_on_background(
    background_names_list : list[str],
    background_filename_dictionary : dict[str, dict],
    centering_movement_to_undo: list[float],
    list_of_bounds : list[tuple[float, float, float, float]],
    list_of_colors : list[str]
    ):
    gpd.options.display_precision = 6
    lines : list[LineString] = []
    f, ax = plt.subplots(figsize=(32, 24))
    undo_centering(background_names_list, lines, centering_movement_to_undo, background_filename_dictionary)
    # print(len(lines))
    frame : gpd.GeoDataFrame = gpd.GeoDataFrame(data={'geometry': lines}, crs="EPSG:4326")
    frame.plot(aspect=1, color='blue', ax=ax, label='background')
    for (i, bound) in enumerate(list_of_bounds):
        # print(f"adding bound {bound}")
        minx, miny, maxx, maxy = bound
        # maxx += centering_movement_to_undo[0]
        # minx += centering_movement_to_undo[0]
        # maxy += centering_movement_to_undo[1]
        # miny += centering_movement_to_undo[1]
        xy = (minx, miny)
        width = maxx - minx
        height = maxy - miny
        # print(f"xy = {xy}, width = {width}, height = {height}")
        ax.add_patch(Rectangle(xy=xy, width=width, height=height, alpha=1, fill=False, edgecolor=list_of_colors[i],
            linewidth=2))
    ax.legend()
    plt.show()