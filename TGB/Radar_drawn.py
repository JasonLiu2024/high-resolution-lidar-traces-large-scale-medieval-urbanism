from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Metric import Close_Points, Kappa_Binary
from numpy.typing import NDArray
from copy import deepcopy
from typing import Callable

class Compare:
    """a is truth, b is prediction"""
    def __init__(self, name : str, mask_a : NDArray, mask_b : NDArray,
    fatten_mask_radius : int=0,
    background : NDArray=None, mask_a_name : str='mask_a_name', mask_b_name : str='mask_b_name', background_name : str='background'):
        self.name = name
        self.channel = 3
        assert mask_a.shape == mask_b.shape, "masks no line up"
        if background:
            assert mask_a.shape[0:2] == background.shape[0:2], "background dims bad"
            self.background = background
        else:
            self.background = np.ones(shape=(mask_a.shape[0], mask_a.shape[1], self.channel)) * 255
        self.mask_a = mask_a
        self.mask_b = mask_b
        if fatten_mask_radius > 0:
            _, _, mask_a_fatten = Close_Points(self.mask_a, self.mask_b, fatten_mask_radius)
            self.mask_b *= mask_a_fatten 
        self.mask_a_name = mask_a_name
        self.mask_b_name = mask_b_name
        self.background_name = background_name
        self.x = self.mask_a.shape[0]
        self.y = self.mask_a.shape[1]

    def draw_lines_on_image(self, lines_mask, filename : str=None):
        if not filename:
            img = deepcopy(self.background)
        else:     
            img = np.asarray((Image.open(filename).convert('RGB')))
        x_length, y_length, _ = img.shape
        # print(f"image shape: {img.shape}")
        for x in range(x_length):
            for y in range(y_length):
                if lines_mask[x][y] == 1:
                    img[x][y] = [0, 0, 0]
        return img

    def statistics(self, show_other : bool, save : bool=False, square : bool=True, 
            radius : int=10, figsize : tuple[int, int]=(16, 12), DPI : int=400, illustrate_process : bool=False):
        f, ax = plt.subplots(nrows=3 if illustrate_process else 2, ncols=2, figsize=figsize)
        r = radius * 2 + 1 # displaying square for size reference, add center pixel
        result_filename = f'{self.name}_radius={radius}_result.png'
        mask_a_on_image = self.draw_lines_on_image(self.mask_a, None)
        ax[0, 0].imshow((mask_a_on_image).astype(np.uint8))
        ax[0, 0].set_title(f"{self.mask_a_name}")
        ax[0, 0].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        mask_b_on_image = self.draw_lines_on_image(self.mask_b, None)
        ax[0, 1].imshow((mask_b_on_image).astype(np.uint8))
        ax[0, 1].set_title(f"{self.mask_b_name}")
        ax[0, 1].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        # one direction
        print(f"{self.mask_a_name} to {self.mask_b_name}")
        a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        color_hits = np.ones((self.x, self.y, self.channel)) * 255
        color_hits += np.stack((a_to_b_hits * -255, a_to_b_hits * -255, a_to_b_hits * 0), axis=2)
        didnt_hits = self.mask_a - a_to_b_hits
        color_hits += np.stack((didnt_hits * 0, didnt_hits * -255, didnt_hits * -255), axis=2)
        if square:
            s = np.ones((r, r))
            s_color = np.stack((s * 0, s * 0, s * 255), axis=2)
            color_hits[0:r, 0:r, :] = s_color
        color_hits = np.clip(a=color_hits, a_min=0, a_max=255) # clip values
        ax[1, 0].imshow(color_hits.astype(np.uint8))
        if show_other == True:
            ax[1, 0].imshow((np.stack([(1 - self.mask_b),
                                     (1 - self.mask_b),
                                     (1 - self.mask_b)], axis=2) * 255).astype(np.uint8), alpha=0.5)
        ax[1, 0].set_title(f"{self.mask_a_name} pixels colored by hits (blue {a_to_b_score * 100:.2f}%)")
        ax[1, 0].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        # other direction
        print(f"{self.mask_a_name} to {self.mask_b_name}")
        b_to_a_score, b_to_a_hits, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        color_hits = np.ones((self.x, self.y, self.channel)) * 255
        color_hits += np.stack((b_to_a_hits * -255, b_to_a_hits * -255, b_to_a_hits * 0), axis=2)
        didnt_hits = self.mask_b - b_to_a_hits
        color_hits += np.stack((didnt_hits * 0, didnt_hits * -255, didnt_hits * -255), axis=2)
        if square:
            s = np.ones((r, r))
            s_color = np.stack((s * 0, s * 0, s * 255), axis=2)
            color_hits[0:r, 0:r, :] = s_color
        ax[1, 1].imshow(color_hits.astype(np.uint8))
        if show_other == True:
            ax[1, 1].imshow((np.stack([(1 - self.mask_a),
                                     (1 - self.mask_a),
                                     (1 - self.mask_a)], axis=2) * 255).astype(np.uint8), alpha=0.5)
        ax[1, 1].set_title(f"{self.mask_b_name} pixels colored by hits (blue {b_to_a_score * 100:.2f}%)")
        ax[1, 1].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        """confusion matrix for binary classification"""
        # a = truth, b = prediction
        #           on a         NOT on a
        # on b      true pos     false pos
        # NOT on b  false neg    true neg
        true_positive = np.sum(a_fatten * b_fatten)
        false_positive = np.sum(b_fatten) - true_positive
        false_negative = np.sum(a_fatten) - true_positive
        true_negative = self.mask_a.size - (true_positive + false_positive + false_negative)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(a_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(b_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(a_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(b_fatten)
        # kappa coefficient (Cohen's Kappa): (Po - Pe)/(1 - Pe)
        # Po: (true pos + true neg)/everything
        # Pyes: (all positive truths * all positive predictions)/everything
        # Pno: (all negative truths * all negative predictions)/everything
        # Pe: Pyes + Pno
        kappa = Kappa_Binary(true_positive=true_positive, false_positive=false_positive, false_negative=false_negative, true_negative=true_negative)
        confusion_matrix = {}
        confusion_matrix['true positive']       = true_positive
        confusion_matrix['false positive']      = false_positive
        confusion_matrix['false negative']      = false_negative
        confusion_matrix['true negative']       = true_negative
        confusion_matrix['error of omission']   = error_of_omission
        confusion_matrix['error of comission']  = error_of_comission
        confusion_matrix['user accuracy']       = users_accuracy
        confusion_matrix['producer accuracy']   = producers_accuracy
        confusion_matrix['kappa']               = kappa

        if illustrate_process:
            tiling_shape = (self.x, self.y, 1)
            true_positive_display = np.expand_dims(a=a_fatten * b_fatten, axis=2) * np.tile(
                A=np.asarray([0, 255, 0]), reps=tiling_shape) # green
            false_positive_display = np.expand_dims(a=b_fatten - a_fatten * b_fatten, axis=2) * np.tile(
                A=np.asarray([255, 255, 0]), reps=tiling_shape) # yellow
            false_negative_display = np.expand_dims(a=a_fatten - a_fatten * b_fatten, axis=2) * np.tile(
                A=np.asarray([255, 20, 147]), reps=tiling_shape) # pink
            # true negative displayed as black
            ax[2, 0].imshow((true_positive_display + false_positive_display + false_negative_display).astype(np.uint8))
            ax[2, 0].set_title(f"confusion matrix display ({self.mask_a.size} pixels total)")
            ax[2, 0].tick_params(left=False, right=False, labelleft=False , 
                labelbottom=False, bottom=False)

            confusion_matrix_display = ax[2, 1].table(
                cellText=[[f"(green) {true_positive}", f"(yellow) {false_positive}"], 
                        [f"(deep pink) {false_negative}", f"(black) {true_negative}"]],
                rowLabels=["on crest line", "NOT on crest line"],
                colLabels=["on drawn line", "NOT on drawn line"],
                loc='center'
            )
            confusion_matrix_display.auto_set_column_width(col=[0, 1])
            ax[2, 1].set_axis_off()

        # show and save image
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        if save == True:
            current_figure.savefig(result_filename, dpi=DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        return a_to_b_score, b_to_a_score, result_filename, confusion_matrix

from scipy import ndimage
class Comparer:
    """a is truth, b is prediction"""
    def __init__(self, name : str, mask_a : NDArray, mask_b : NDArray,
    names : dict[str, str], result_filename : str,
    fatten_mask_radius : int=0, dir : str="", filetype : str="png", DPI : int=100,
    background : NDArray=None, background_name : str='background',
    ):
        self.name = name
        self.channel = 3
        assert mask_a.shape == mask_b.shape, "masks no line up"
        if background is not None:
            assert mask_a.shape[0:2] == background.shape[0:2], "background dims bad"
            self.background = background
        else:
            self.background = np.ones(shape=(mask_a.shape[0], mask_a.shape[1], self.channel)) * 255
        self.mask_a = mask_a
        self.mask_b = mask_b
        # if fatten_mask_radius > 0:
        #     _, _, mask_a_fatten = Close_Points(self.mask_a, self.mask_b, fatten_mask_radius)
        #     self.mask_b *= mask_a_fatten 
        self.background_name = background_name # unused
        self.x = self.mask_a.shape[0]
        self.y = self.mask_a.shape[1]
        self.DPI = DPI
        self.filetype = filetype
        self.dir = dir
        self.filename = result_filename

    def centerline_over_background(self, background : NDArray, color = [0, 0, 255], 
            figsize : tuple[int, int]=(16, 16)):
        background[self.mask_a == 1] = color
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(background)
        current_figure = plt.gcf()
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        plt.show()
        plt.draw()
        filename = f"radar_{self.filename}_over_background.{self.filetype}"
        print(f"file: {filename}")
        current_figure.savefig(filename, dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')

    def detect_a_to_b(self, radius : int=10, figsize : tuple[int, int]=(16, 16), DPI : int=400,
        hit_color = [0, 0, 255], didnt_hit_color = [255, 0, 0]):
        _, a_to_b_hits, _ = Close_Points(self.mask_a, self.mask_b, radius)
        # b_to_a_score, b_to_a_hits, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        """A to B hits"""
        f, ax = plt.subplots(figsize=figsize)
        a_hits = np.ones(shape=(self.x, self.y, 3)) * 255
        a_hits[self.mask_a == 1] = hit_color # add hits
        a_hits[self.mask_a - a_to_b_hits == 1] = didnt_hit_color # add didn't-hits
        a_hits[:radius * 2 + 1, :radius * 2 + 1] = didnt_hit_color
        ax.imshow(a_hits.astype(np.uint8))
        current_figure = plt.gcf()
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        plt.show()
        plt.draw()
        filename = f"{self.dir}radar_{self.filename}_r={radius}px.{self.filetype}"
        print(f"file: {filename}")
        current_figure.savefig(filename, dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')

    def detect_b_to_a(self, radius : int=10, figsize : tuple[int, int]=(16, 16), DPI : int=400,
        hit_color = [0, 0, 255], didnt_hit_color = [255, 0, 0]):
        # a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        _, b_to_a_hits, _ = Close_Points(self.mask_b, self.mask_a, radius)
        """A to B hits"""
        f, ax = plt.subplots(figsize=figsize)
        b_hits = np.ones(shape=(self.x, self.y, 3)) * 255
        b_hits[self.mask_b == 1] = hit_color # add hits
        b_hits[self.mask_b - b_to_a_hits == 1] = didnt_hit_color # add didn't-hits
        b_hits[:radius * 2 + 1, :radius * 2 + 1] = didnt_hit_color
        b_hits[:radius * 2 + 1, :radius * 2 + 1] = didnt_hit_color
        ax.imshow(b_hits.astype(np.uint8))
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        filename = f"{self.dir}crest_line_{self.filename}.{self.filetype}"
        print(f"file: {filename}")
        current_figure.savefig(filename, dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        # plt.clf()

    def detect_a_to_b_thicken(self, radius : int=10, figsize : tuple[int, int]=(16, 16), DPI : int=400,
        hit_color = [0, 0, 255], didnt_hit_color = [255, 0, 0]):
        # a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        a_to_b_score, a_to_b_hits, b_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        hits_dilated = ndimage.binary_dilation(input=self.mask_a).astype(np.uint8)
        a_to_b_hits_dilated = ndimage.binary_dilation(input=a_to_b_hits).astype(np.uint8)
        didnt_hits = hits_dilated - a_to_b_hits_dilated

        """A to B hits"""
        f, ax = plt.subplots(figsize=figsize)
        a_hits = np.ones(shape=(self.x, self.y, 3)) * 255
        a_hits[hits_dilated == 1] = hit_color # add hits
        a_hits[didnt_hits == 1] = didnt_hit_color # add didn't-hits
        ax.imshow(a_hits.astype(np.uint8))
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        current_figure.savefig(f"{self.dir}radar_thicken_{self.filename}.{self.filetype}", dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        plt.clf()

    def detect_b_to_a_thicken(self, radius : int=10, figsize : tuple[int, int]=(16, 16), DPI : int=400,
        hit_color = [0, 0, 255], didnt_hit_color = [255, 0, 0]):
        # a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        b_to_a_score, b_to_a_hits, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        hits_dilated = ndimage.binary_dilation(input=self.mask_b).astype(np.uint8)
        b_to_a_hits_dilated = ndimage.binary_dilation(input=b_to_a_hits).astype(np.uint8)
        didnt_hits = hits_dilated - b_to_a_hits_dilated

        """B to A hits"""
        f, ax = plt.subplots(figsize=figsize)
        b_hits = np.ones(shape=(self.x, self.y, 3)) * 255
        b_hits[hits_dilated == 1] = hit_color # add hits
        b_hits[didnt_hits == 1] = didnt_hit_color # add didn't-hits
        b_hits[:radius * 2 + 1, :radius * 2 + 1] = didnt_hit_color
        ax.imshow(b_hits.astype(np.uint8))
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        current_figure.savefig(f"{self.dir}crest_line_thicken_{self.filename}.{self.filetype}", dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        plt.clf()

    def crest_lines(self,):
        print("NOTE: dilated by 1 pixel to display more clearly") 
        f, ax = plt.subplots(figsize=(16, 16))
        b = ndimage.binary_dilation(input=self.mask_b).astype(np.uint8)
        ax.imshow(1 - b, cmap='gray')
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        current_figure.savefig(f"{self.dir}crest_lines_{self.filename}.{self.filetype}", dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        plt.clf()

    def GPR_lines(self,):
        f, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(1 - self.mask_a.astype(np.uint8), cmap='gray')
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        current_figure.savefig(f"{self.dir}GPR_lines_{self.filename}.{self.filetype}", dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        plt.clf()

    def GPR_image(self, figsize : tuple[int, int]=(16, 16), DPI : int=400,):
        """Just image"""
        f, ax = plt.subplots(figsize=(16, 16))
        # GPR_image = self.draw_lines_on_image(lines_mask=None, filename='25~89.png_need_ridges.png')
        ax.imshow(self.background.astype(np.uint8))
        ax.tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        current_figure.savefig(f"{self.dir}GPR_{self.filename}.{self.filetype}", dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        plt.clf()

    def draw_lines_on_image(self, lines_mask, filename : str=None):
        if not filename:
            img = deepcopy(self.background)
        else:     
            img = np.asarray((Image.open(filename).convert('RGB')))
        x_length, y_length, _ = img.shape
        # print(f"image shape: {img.shape}")
        if lines_mask is not None:
            for x in range(x_length):
                for y in range(y_length):
                    if lines_mask[x][y] == 1:
                        img[x][y] = [0, 0, 0]
        return img
    
    def CM(self, 
            true_pos_color : NDArray, true_neg_color : NDArray, 
            false_pos_color : NDArray, false_neg_color : NDArray, background_color : NDArray,
            comparison_space : NDArray,   
            save : bool=False, 
            radius : int=10, figsize : tuple[int, int]=(16, 16), DPI : int=600, filetype='tiff'
            ):
        r = radius * 2 + 1 # displaying square for size reference, add center pixel
        print(f"detector diameter: {r} (radius {radius})")
        a_to_b_score, _, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        b_to_a_score, _, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        """confusion matrix for binary classification"""
        # a = truth, b = prediction
        #           on a         NOT on a
        # on b      true pos     false pos
        # NOT on b  false neg    true neg
        a_fatten *= comparison_space # trim away excess
        b_fatten *= comparison_space
        true_positive = (a_fatten & b_fatten).astype(np.uint8)
        true_positive_ct = np.sum(true_positive)
        false_positive = b_fatten - true_positive
        false_positive_ct = np.sum(false_positive)
        false_negative = a_fatten - true_positive
        false_negative_ct = np.sum(false_negative)
        true_negative = comparison_space - true_positive - false_positive - false_negative
        true_negative_ct = np.sum(true_negative)
        print(f"Sum: {true_positive_ct} + {false_positive_ct} + {false_negative_ct} + {true_negative_ct} \
            = TP.{true_positive_ct} + FP.{false_positive_ct} + FN.{false_negative_ct} + TN{true_negative_ct} Should be {np.sum(comparison_space)}")
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative_ct/np.sum(a_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive_ct/np.sum(b_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive_ct/np.sum(a_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive_ct/np.sum(b_fatten)
        # kappa coefficient (Cohen's Kappa): (Po - Pe)/(1 - Pe)
        # Po: (true pos + true neg)/everything
        # Pyes: (all positive truths * all positive predictions)/everything
        # Pno: (all negative truths * all negative predictions)/everything
        # Pe: Pyes + Pno
        # 
        F1 = true_positive_ct / (true_positive_ct + (false_positive_ct + false_negative_ct) / 2.0)
        # kappa = Kappa_Binary(true_positive=true_positive_ct, false_positive=false_positive_ct, 
        #     false_negative=false_negative_ct, true_negative=true_negative_ct)
        print(f"statistics calculated")
        print(f"Error of Omission:   {error_of_omission}")
        print(f"Error of Commission: {error_of_comission}")
        print(f"User Accuracy:       {users_accuracy}")
        print(f"Producer Accuracy:   {producers_accuracy}")
        print(f"F1: {F1}")
        # print(f"Kappa: {kappa}")
        # outdated version
        tiling_shape = (self.x, self.y, 1)
        # mass-indexing is concise but slow
        cm_display = np.tile(A=background_color, reps=tiling_shape)
        cm_display[true_positive == 1] = true_pos_color
        cm_display[false_positive == 1] = false_pos_color
        # print(f"check: true positive & false positive: {np.sum(true_positive * false_positive)}")
        cm_display[false_negative == 1] = false_neg_color
        # print(f"check: false positive & false negative: {np.sum(false_positive * false_negative)}")
        cm_display[true_negative == 1] = true_neg_color
        # print(f"check: true positive & true_negative: {np.sum(true_positive * true_negative)}")
        print(cm_display.shape)
        # if trimmer is not None:
        #     cm_display[trimmer == 0] = true_neg_color
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow((cm_display).astype(np.uint8))
        ax.tick_params(left=False, right=False, labelleft=False, 
            labelbottom=False, bottom=False)

        # show and save image
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        if save == True:
            filename = f"{self.dir}confusion_matrix_{self.filename}.{filetype}"
            print(f"File: {filename}")
            current_figure.savefig(filename, dpi=self.DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        return cm_display, a_to_b_score, b_to_a_score
   

class Compare_Paper:
    """a is truth, b is prediction"""
    def __init__(self, name : str, mask_a : NDArray, mask_b : NDArray,
    names : dict[str, str], result_filename : str,
    fatten_mask_radius : int=0,
    background : NDArray=None, background_name : str='background',
    ):
        self.name = name
        self.channel = 3
        assert mask_a.shape == mask_b.shape, "masks no line up"
        if background is not None:
            assert mask_a.shape[0:2] == background.shape[0:2], "background dims bad"
            self.background = background
        else:
            self.background = np.ones(shape=(mask_a.shape[0], mask_a.shape[1], self.channel)) * 255
        self.mask_a = mask_a
        self.mask_b = mask_b
        if fatten_mask_radius > 0:
            _, _, mask_a_fatten = Close_Points(self.mask_a, self.mask_b, fatten_mask_radius)
            self.mask_b *= mask_a_fatten 
        self.background_name = background_name # unused
        self.x = self.mask_a.shape[0]
        self.y = self.mask_a.shape[1]
        self.TL_name = names['TL']
        self.TR_name = names['TR']
        self.BL_name = names['BL']
        self.BR_name = names['BR']
        self.filename = result_filename

    def draw_lines_on_image(self, lines_mask, filename : str=None):
        if not filename:
            img = deepcopy(self.background)
        else:     
            img = np.asarray((Image.open(filename).convert('RGB')))
        x_length, y_length, _ = img.shape
        # print(f"image shape: {img.shape}")
        if lines_mask is not None:
            for x in range(x_length):
                for y in range(y_length):
                    if lines_mask[x][y] == 1:
                        img[x][y] = [0, 0, 0]
        return img

    def statistics(self, show_other : bool, save : bool=False, square : bool=True, 
            radius : int=10, figsize : tuple[int, int]=(16, 12), DPI : int=400, illustrate_process : bool=False):
        f, ax = plt.subplots(nrows=3 if illustrate_process else 2, ncols=2, figsize=figsize)
        r = radius * 2 + 1 # displaying square for size reference, add center pixel
        """Just image"""
        # GPR_image = self.draw_lines_on_image(lines_mask=None, filename='25~89.png_need_ridges.png')
        ax[0, 0].imshow(self.background.astype(np.uint8))
        ax[0, 0].set_title(self.TL_name)
        ax[0, 0].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        """Just radar"""
        # mask_a_on_image = self.draw_lines_on_image(self.mask_a, None)
        ax[1, 0].imshow((1 - self.mask_a).astype(np.uint8), cmap='gray')
        ax[1, 0].set_title(self.BL_name)
        ax[1, 0].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        """Just GPR"""
        # mask_b_on_image = self.draw_lines_on_image(self.mask_b, None)
        ax[0, 1].imshow((1 - self.mask_b).astype(np.uint8), cmap='gray')
        ax[0, 1].set_title(self.TR_name)
        ax[0, 1].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        b_to_a_score, b_to_a_hits, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        """confusion matrix for binary classification"""
        # a = truth, b = prediction
        #           on a         NOT on a
        # on b      true pos     false pos
        # NOT on b  false neg    true neg
        true_positive = np.sum(a_fatten * b_fatten)
        false_positive = np.sum(b_fatten) - true_positive
        false_negative = np.sum(a_fatten) - true_positive
        true_negative = self.mask_a.size - (true_positive + false_positive + false_negative)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(a_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(b_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(a_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(b_fatten)
        # kappa coefficient (Cohen's Kappa): (Po - Pe)/(1 - Pe)
        # Po: (true pos + true neg)/everything
        # Pyes: (all positive truths * all positive predictions)/everything
        # Pno: (all negative truths * all negative predictions)/everything
        # Pe: Pyes + Pno
        kappa = Kappa_Binary(true_positive=true_positive, false_positive=false_positive, false_negative=false_negative, true_negative=true_negative)
        confusion_matrix = {}
        confusion_matrix['true positive']       = true_positive
        confusion_matrix['false positive']      = false_positive
        confusion_matrix['false negative']      = false_negative
        confusion_matrix['true negative']       = true_negative
        confusion_matrix['error of omission']   = error_of_omission
        confusion_matrix['error of comission']  = error_of_comission
        confusion_matrix['user accuracy']       = users_accuracy
        confusion_matrix['producer accuracy']   = producers_accuracy
        confusion_matrix['kappa']               = kappa

        tiling_shape = (self.x, self.y, 1)
        true_positive_display = np.expand_dims(a=a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([0, 255, 0]), reps=tiling_shape) # green
        false_positive_display = np.expand_dims(a=b_fatten - a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([255, 255, 0]), reps=tiling_shape) # yellow
        false_negative_display = np.expand_dims(a=a_fatten - a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([0, 255, 255]), reps=tiling_shape) # blue
        # true negative displayed as black
        ax[1, 1].imshow((true_positive_display + false_positive_display + false_negative_display).astype(np.uint8))
        ax[1, 1].set_title(self.BR_name)
        ax[1, 1].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        # confusion_matrix_display = ax[2, 1].table(
        #     cellText=[[f"(green) {true_positive}", f"(yellow) {false_positive}"], 
        #             [f"(deep pink) {false_negative}", f"(black) {true_negative}"]],
        #     rowLabels=["on crest line", "NOT on crest line"],
        #     colLabels=["on drawn line", "NOT on drawn line"],
        #     loc='center'
        # )
        # confusion_matrix_display.auto_set_column_width(col=[0, 1])
        # ax[2, 1].set_axis_off()

        # show and save image
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        if save == True:
            current_figure.savefig(self.filename, dpi=DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        return a_to_b_score, b_to_a_score, confusion_matrix
    
class Compare_Paper_TGB:
    """a is truth, b is prediction"""
    def __init__(self, name : str, mask_a : NDArray, mask_b : NDArray,
    names : dict[str, str], result_filename : str,
    fatten_mask_radius : int=0,
    background : NDArray=None, background_name : str='background',
    ):
        self.name = name
        self.channel = 3
        assert mask_a.shape == mask_b.shape, "masks no line up"
        if background:
            assert mask_a.shape[0:2] == background.shape[0:2], "background dims bad"
            self.background = background
        else:
            self.background = np.ones(shape=(mask_a.shape[0], mask_a.shape[1], self.channel)) * 255
        self.mask_a = mask_a
        self.mask_b = mask_b
        if fatten_mask_radius > 0:
            _, _, mask_a_fatten = Close_Points(self.mask_a, self.mask_b, fatten_mask_radius)
            self.mask_b *= mask_a_fatten 
        self.background_name = background_name # unused
        self.x = self.mask_a.shape[0]
        self.y = self.mask_a.shape[1]
        self.names = names
        self.filename = result_filename

    def draw_lines_on_image(self, lines_mask, filename : str=None):
        if not filename:
            img = deepcopy(self.background)
        else:     
            img = np.asarray((Image.open(filename).convert('RGB')))
        x_length, y_length, _ = img.shape
        # print(f"image shape: {img.shape}")
        if lines_mask is not None:
            for x in range(x_length):
                for y in range(y_length):
                    if lines_mask[x][y] == 1:
                        img[x][y] = [0, 0, 0]
        return img

    def statistics(self, show_other : bool, save : bool=False, square : bool=True, 
            radius : int=10, figsize : tuple[int, int]=(16, 12), DPI : int=400, illustrate_process : bool=False):
        f, ax = plt.subplots(nrows=3, figsize=figsize)
        r = radius * 2 + 1 # displaying square for size reference, add center pixel
        """Just crest lines"""
        mask_a_on_image = self.draw_lines_on_image(self.mask_a, None)
        ax[0].imshow((mask_a_on_image).astype(np.uint8))
        ax[0].set_title(self.names['1'])
        ax[0].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        """Just GPR"""
        mask_b_on_image = self.draw_lines_on_image(self.mask_b, None)
        ax[1].imshow((mask_b_on_image).astype(np.uint8))
        ax[1].set_title(self.names['2'])
        ax[1].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)
        a_to_b_score, a_to_b_hits, a_fatten = Close_Points(self.mask_a, self.mask_b, radius)
        b_to_a_score, b_to_a_hits, b_fatten = Close_Points(self.mask_b, self.mask_a, radius)
        """confusion matrix for binary classification"""
        # a = truth, b = prediction
        #           on a         NOT on a
        # on b      true pos     false pos
        # NOT on b  false neg    true neg
        true_positive = np.sum(a_fatten * b_fatten)
        false_positive = np.sum(b_fatten) - true_positive
        false_negative = np.sum(a_fatten) - true_positive
        true_negative = self.mask_a.size - (true_positive + false_positive + false_negative)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(a_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(b_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(a_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(b_fatten)
        # kappa coefficient (Cohen's Kappa): (Po - Pe)/(1 - Pe)
        # Po: (true pos + true neg)/everything
        # Pyes: (all positive truths * all positive predictions)/everything
        # Pno: (all negative truths * all negative predictions)/everything
        # Pe: Pyes + Pno
        kappa = Kappa_Binary(true_positive=true_positive, false_positive=false_positive, false_negative=false_negative, true_negative=true_negative)
        confusion_matrix = {}
        confusion_matrix['true positive']       = true_positive
        confusion_matrix['false positive']      = false_positive
        confusion_matrix['false negative']      = false_negative
        confusion_matrix['true negative']       = true_negative
        confusion_matrix['error of omission']   = error_of_omission
        confusion_matrix['error of comission']  = error_of_comission
        confusion_matrix['user accuracy']       = users_accuracy
        confusion_matrix['producer accuracy']   = producers_accuracy
        confusion_matrix['kappa']               = kappa

        tiling_shape = (self.x, self.y, 1)
        true_positive_display = np.expand_dims(a=a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([0, 0, 255]), reps=tiling_shape) # green
        false_positive_display = np.expand_dims(a=b_fatten - a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([255, 165, 0]), reps=tiling_shape) # yellow
        false_negative_display = np.expand_dims(a=a_fatten - a_fatten * b_fatten, axis=2) * np.tile(
            A=np.asarray([128, 128, 128]), reps=tiling_shape) # blue
        # true negative displayed as black
        ax[2].imshow((true_positive_display + false_positive_display + false_negative_display).astype(np.uint8))
        ax[2].set_title(self.names['3'])
        ax[2].tick_params(left=False, right=False, labelleft=False , 
            labelbottom=False, bottom=False)

        # show and save image
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        if save == True:
            current_figure.savefig(self.filename, dpi=DPI, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        return a_to_b_score, b_to_a_score, confusion_matrix