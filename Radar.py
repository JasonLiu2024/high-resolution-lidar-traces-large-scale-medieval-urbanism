from PIL import Image
import numpy as np
from DualColor import Grayscale, Threshold
import cv2 as cv
from skimage.filters import meijering
from PIL import ImageOps
from Display import Plot_2D
import matplotlib.pyplot as plt
from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params, load_json
import json
from Metric import Close_Points, Kappa_Binary
from numpy.typing import NDArray
# def save_image_needs_ridges(self):
#     """saving square image, to work with ridge-detection package
#         image name: need_ridges.png"""
#     plotter = Plot_2D(init_color='black', warp_color='white', thickness=self.thin)
#     f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
#     # print(f"image is {self.x_length}, {self.y_length}")
#     ax.set_xlim(0, self.x_length)
#     ax.set_ylim(0, self.y_length)
#     plotter.Plot_Background(f, ax, self.img_display, "BLACK")
#     plotter.Plot_ImageTransform(f, ax, self.img_display)
#     # plotter.Plot_Labels(f, ax)
#     plt.imshow(np.flip(self.mask, axis=0) * 255, alpha=(1 - np.flip(self.mask, axis=0)), cmap='gray')
#     ax.autoscale()
#     current_figure = plt.gcf()
#     plt.show()
#     plt.draw()
#     plotter.Plot_SaveTrim(f, ax, current_figure, 'need_ridges.png', 
#     (0, 0), (x_length, y_length)) # save all

class Compare:
    def __init__(self, name : str, mask : NDArray,
        crop_ll=None, crop_lr=None):
        # print("hey")
        self.name = f"{name}.png"
        self.mask = mask
        self.img = Image.open(self.name)
        self.y_length, self.x_length = self.img.size
        blank_array = np.zeros(shape=(self.y_length, self.x_length, 3))
        # threshold_value, img_thresholded = Threshold(Grayscale(imageFile, "WHITE"))
        self.img_grayscale = Grayscale(self.img, "WHITE")
        # (255 - np.asarray(img_grayscale)) # <- numpy alternative to invert black and white
        self.inverted = ImageOps.invert(self.img_grayscale)
        self.img_display = np.flip(self.inverted, axis=0)
        self.thic = 1.6
        self.thin = 0.1
        if crop_ll == None:
            self.ll = (0, 0)
        else:
            self.ll = crop_ll
        if crop_lr == None:
            self.lr = (self.y_length, self.y_length)
        else:
            self.lr = crop_lr
        # need to call-initialize
        self.mask_radar_ridges = None
        self.mask_crestlines = None
        self.radar_ridges_over_radar = None
    def gray(self):
        """grayscale PIL Image"""
        return self.img_grayscale
    def ridges_skimage(self):
        """skimage ridges (meijering)"""
        sigmas = [1, 2, 3, 4]
        black_ridges = True
        img_ridges_skimage = meijering(np.asarray(self.img_grayscale), black_ridges=black_ridges, sigmas=sigmas)
        # plt.matshow(img_ridges_skimage, cmap=plt.cm.gray)
        return img_ridges_skimage
    def ridges_cv2(self):
        """cv2 ridges"""
        # MUST use: opencv-contrib-python, NOT opencv-python
        ridge_filter = cv.ximgproc.RidgeDetectionFilter_create()
        # input must be RGB
        img_ridges_cv = ridge_filter.getRidgeFilteredImage(np.asarray(self.img.convert('RGB')))
        # plt.matshow(img_ridges_cv, cmap=plt.cm.gray)
        return img_ridges_cv
    def save_lines(self, warped_vertices, crestline_edges): 
        """saving square image, to work with ridge-detection package
            image name: lines.png
            NOTE: saves FULL picture, NOT crop!"""
        # print(f"yo")
        # use plotter object!
        plotter = Plot_2D(init_color='black', warp_color='white', thickness=self.thin)
        f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
        # print(f"image is {self.x_length}, {self.y_length}")
        ax.set_xlim(0, self.x_length)
        ax.set_ylim(0, self.y_length)
        plotter.Plot_Background(f, ax, self.img_display, "BLACK")
        plotter.Plot_Crestlines_Warp(f, ax, warped_vertices, crestline_edges)
        # plotter.Plot_Labels(f, ax)
        # plotter.Plot_Background(f, ax, self.img_display, "RED")
        # print("paint stuff red")
        # plt.imshow(np.flip(self.mask, axis=0) * 255, alpha=(1 - np.flip(self.mask, axis=0)), cmap='gray')
        # print("Compare::save_lines::drew mask!")
        ax.autoscale()
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        plotter.Plot_SaveTrim(f, ax, current_figure, 'lines_full.png', 
            (0, 0), (self.y_length, self.x_length)) # save all
        """why am I re-doing this? matplotlib always adds lines AFTER the mask!"""
        plotter = Plot_2D(init_color='black', warp_color='white', thickness=self.thin)
        f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
        # print(f"image is {self.x_length}, {self.y_length}")
        ax.set_xlim(0, self.x_length)
        ax.set_ylim(0, self.y_length)
        # plotter.Plot_Background(f, ax, self.img_display, "BLACK")
        # plotter.Plot_Crestlines_Warp(f, ax, warped_vertices, crestline_edges)
        # plotter.Plot_Labels(f, ax)
        # plotter.Plot_Background(f, ax, self.img_display, "RED")
        # print("paint stuff red")
        plt.imshow(np.flip(Image.open('lines_full.png'), axis=0))
        plt.imshow(np.flip(self.mask, axis=0) * 255, alpha=(1 - np.flip(self.mask, axis=0)), cmap='gray')
        ax.autoscale()
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        plotter.Plot_SaveTrim(f, ax, current_figure, 'lines.png', 
            self.ll, self.lr) # save all
    def save_image_needs_ridges(self):
        """saving square image, to work with ridge-detection package
            image name: need_ridges.png"""
        plotter = Plot_2D(init_color='black', warp_color='white', thickness=self.thin)
        f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
        ax.set_xlim(0, self.x_length)
        ax.set_ylim(0, self.y_length)
        plotter.Plot_Background(f, ax, self.img_display, "BLACK")
        plotter.Plot_ImageTransform(f, ax, self.img_display)
        # plotter.Plot_Labels(f, ax)
        # plt.imshow(np.flip(self.mask, axis=0) * 255, alpha=(1 - np.flip(self.mask, axis=0)), cmap='gray')
        ax.autoscale()
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        plotter.Plot_SaveTrim(f, ax, current_figure, f'{self.name}_need_ridges.png', 
            self.ll, self.lr) # save all
    # def save_image_with_crestlines(self, crestline_vertices, crestline_edges):
    #     """saving square image, to work with ridge-detection package
    #         image name: image_with_crestlines.png"""
    #     print("save_image_with_crestlines")
    #     plotter = Plot_2D(init_color='black', warp_color='red', thickness=self.thic)
    #     f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
    #     # print(f"image is {self.x_length}, {self.y_length}")
    #     ax.set_xlim(0, self.x_length)
    #     ax.set_ylim(0, self.y_length)
    #     img = np.asarray((Image.open('need_ridges.png').convert('RGB')))
    #     # ax.imshow(np.flip(Image.open('need_ridges.png'), axis=0))
    #     x_length, y_length, _ = img.shape
    #     for x in range(x_length):
    #         for y in range(y_length):
    #             if self.mask_crestlines[x][y] == 1:
    #                 img[x][y] = [255, 0, 0]
    #     ax.imshow(np.flip(img, axis=0))
    #     # ax.imshow(np.flip(np.stack((self.mask_crestlines * 255, self.mask_crestlines * 0, self.mask_crestlines * 0), axis=2), axis=0), alpha=np.flip(self.mask_crestlines, axis=1))
    #     # plotter.Plot_Crestlines_Warp(f, ax, crestline_vertices, crestline_edges)
    #     # plotter.Plot_Labels(f, ax)
    #     # plt.imshow(np.flip(self.mask, axis=0) * 255, alpha=(1 - np.flip(self.mask, axis=0)), cmap='gray')
    #     ax.autoscale()
    #     current_figure = plt.gcf()
    #     plt.show()
    #     plt.draw()
    #     plotter.Plot_SaveAxes(f, ax, current_figure, filename='image_with_crestlines.png')
    def get_mask_image_ridges(self):
        """in the output:
            red lines:   the ridge/centerline of the line
            green lines: the edge/boundary/contour of the line
            blue dots:   the junction of a line (where two or more lines meet)"""
        print(f"Processor::get_mask_image_ridges")
        # get_ridges_from_this = 'need_ridges.png'
        get_ridges_from_this = f'{self.name}_need_ridges.png'
        d = {
            "path_to_file": get_ridges_from_this,
            "mandatory_parameters": {
                # how much detail to identify
                "Sigma": 5.4, # 3.39, estimated by Line_width
                "Lower_Threshold": 0.34, # 0.34, estimated by High_contrast
                "Upper_Threshold": 1.02, # 1.02, estimated by Low_contrast
                "Maximum_Line_Length": 0, # 0
                "Minimum_Line_Length": 0, # 0
                "Darkline": "LIGHT", # LIGHT = lines are lighter pixels
                "Overlap_resolution": "NONE"
            },

            "optional_parameters": {
                "Line_width": 100.0, # 10.0
                "High_contrast": 200, # 200
                "Low_contrast": 80 # 80
            },

            "further_options": {
                "Correct_position": True,
                "Estimate_width": True,
                "doExtendLine": True,
                "Show_junction_points": True,
                "Show_IDs": False,
                "Display_results": True,
                "Preview": False, # True
                "Make_Binary": False,
                "save_on_disk": True
            }
        }
        config_filename = 'config_image_ridges.json'
        with open(config_filename, 'w') as file:
            json.dump(d, file)
        # code below follows the interface of the ridge-detection package
        json_data = load_json(config_filename)
        params = Params(config_filename)
        image = np.asarray(ImageOps.grayscale(Image.open(json_data["path_to_file"])))
        detect = LineDetector(params=config_filename)
        result = detect.detectLines(image) # input: grayscale, proper numpy array
        resultJunction = detect.junctions
        img_all_lines, img_only_lines, img_binary_lines = displayContours_mod(params, result, resultJunction)
        # call-initialize
        self.mask_radar_ridges = img_binary_lines
        self.radar_ridges_over_radar = img_all_lines
        # could be useful!
        return img_all_lines, img_only_lines, img_binary_lines
    # def get_mask_crestlines(self):
    #     # print(f"Processor::get_mask_crestlines")
    #     gray = ImageOps.grayscale(Image.open('lines.png').convert('RGB'))
    #     features_color = [255, 255, 255]
    #     background_color = [0, 0, 0]
    #     from DualColor import binary
    #     mask_crestlines, _ = binary(np.asarray(gray), 0, features_color, background_color)
    #     # call-initialize
    #     self.mask_crestlines = mask_crestlines
    #     # could be useful!
    #     return mask_crestlines
    def draw_lines_on_image(self, lines_mask, filename):
        img = np.asarray((Image.open(filename).convert('RGB')))
        x_length, y_length, _ = img.shape
        print(f"image shape: {img.shape}")
        for x in range(x_length):
            for y in range(y_length):
                if lines_mask[x][y] == 1:
                    img[x][y] = [255, 0, 0]
        return img
    
    def draw(self, lines_mask, hit, other,
        hit_color=np.asarray([0, 0, 255]), no_hit_color=np.asarray([255, 0, 0]),
        background_color=np.asarray([255, 255, 255]), other_color=np.asarray([0, 0, 0])):
        # too memory expensive!
        channels = 3
        x = lines_mask.shape[0]
        y = lines_mask.shape[1]
        # apply background color
        picture = np.tile(background_color, reps=(x, y, 1))
        # applying other
        picture = picture * np.expand_dims(1 - other, 2) # carve out
        other = np.tile(other_color, reps=(x, y, 1))
        other = other * np.expand_dims(other, 2) # carve out all other
        picture = picture + other # draw on picture
        # applying hit
        picture = picture * np.expand_dims(1 - hit, 2) # carve out
        hit = np.tile(hit_color, reps=(x, y, 1))
        hit = hit * np.expand_dims(hit, 2) # carve out all other
        picture = picture + hit # draw on picture
        # applying no-hit
        no_hit = lines_mask - hit
        picture = picture * np.expand_dims(1 - no_hit, 2) # carve out
        no_hit = np.tile(no_hit_color, reps=(x, y, 1))
        no_hit = no_hit * np.expand_dims(no_hit, 2) # carve out all other
        picture = picture + no_hit # draw on picture
        return picture

    def statistics(self, image_lines_mask, crestlines_lines_mask, show_other, save=False, square=True, radius=10,):
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
        r = radius * 2 + 1 # displaying square for size reference, add center pixel
        result_filename = f'{self.name}_radius={radius}_result.png'
        crestlines_over_image = self.draw_lines_on_image(crestlines_lines_mask, ''25~89.png_need_ridges.png'')
        ax[0, 1].imshow(crestlines_over_image)
        ax[0, 1].set_title("warped crestlines over radar image")
        radar_lines_over_image = self.draw_lines_on_image(image_lines_mask, '25~89.png_need_ridges.png')
        # img = np.asarray((Image.open('25~89.png_need_ridges').convert('RGB')))
        # x_length, y_length, _ = img.shape
        # for x in range(x_length):
        #     for y in range(y_length):
        #         if image_lines_mask[x][y] == 1:
        #             img[x][y] = [255, 0, 0]
        ax[0, 0].imshow(radar_lines_over_image)
        ax[0, 0].set_title("image ridges over radar")
        # one direction
        x = image_lines_mask.shape[0]
        y = image_lines_mask.shape[1]
        channel = 3
        print("radar to warped")
        radar_to_warped_score, radar_to_warped_hits, radar_fatten = Close_Points(image_lines_mask, crestlines_lines_mask, radius)
        color_hits = np.ones((x, y, channel)) * 255
        color_hits += np.stack((radar_to_warped_hits * -255, radar_to_warped_hits * -255, radar_to_warped_hits * 0), axis=2)
        didnt_hits = image_lines_mask - radar_to_warped_hits
        color_hits += np.stack((didnt_hits * 0, didnt_hits * -255, didnt_hits * -255), axis=2)
        # pic = self.draw(lines_mask=image_lines_mask, hit=radar_to_warped_hits, other=crestlines_lines_mask)
        # if(show_other):
        #     o_mask = crestlines_lines_mask - image_lines_mask
        #     color_hits += np.stack((o_mask * -255 * 3, o_mask * -255 * 3, o_mask * -255 * 3), axis=2)
        if square:
            s = np.ones((r, r))
            s_color = np.stack((s * 0, s * 0, s * 255), axis=2)
            color_hits[0:r, 0:r, :] = s_color
        color_hits = np.clip(a=color_hits, a_min=0, a_max=255) # clip values
        ax[1, 0].imshow(color_hits.astype(int))
        if show_other == True:
            ax[1, 0].imshow(np.stack([(1 - crestlines_lines_mask),
                                     (1 - crestlines_lines_mask),
                                     (1 - crestlines_lines_mask)], axis=2) * 255, alpha=0.5)
        ax[1, 0].set_title(f"hit (green) vs didn't hit (red): {radar_to_warped_score}")
        # add_B = self.mask_crestlines - self.mask_radar_ridges
        # color_add_B = color_hits + np.stack((add_B * 0, add_B * 0, add_B * 255), axis=2)
        # color_add_B = np.clip(color_add_B, 0, 255) # clip values
        # ax[1, 1].imshow(color_add_B)
        # ax[1, 1].set_title("showing target mask (blue)")
        # the other direction
        print(f"warped to radar")
        warped_to_radar_score, warped_to_radar_hits, warped_fatten = Close_Points(crestlines_lines_mask, image_lines_mask, radius)
        color_hits = np.ones((x, y, channel)) * 255
        color_hits += np.stack((warped_to_radar_hits * -255, warped_to_radar_hits * -255, warped_to_radar_hits * 0), axis=2)
        didnt_hits = crestlines_lines_mask - warped_to_radar_hits
        color_hits += np.stack((didnt_hits * 0, didnt_hits * -255, didnt_hits * -255), axis=2)
        color_hits = np.clip(color_hits, 0, 255) # clip values
        if square:
            s = np.ones((r, r))
            s_color = np.stack((s * 0, s * 0, s * 255), axis=2)
            color_hits[0:r, 0:r, :] = s_color
        ax[1, 1].imshow(color_hits.astype(int))
        if show_other == True:
            ax[1, 1].imshow(np.stack([(1 - image_lines_mask),
                                     (1 - image_lines_mask),
                                     (1 - image_lines_mask)], axis=2) * 255, alpha=0.5)
        ax[1, 1].set_title(f"hit (green) vs didn't hit (red): {warped_to_radar_score}")
        # add_B = self.mask_radar_ridges - self.mask_crestlines
        # color_add_B = color_hits + np.stack((add_B * 0, add_B * 0, add_B * 255), axis=2)
        # color_add_B = np.clip(color_add_B, 0, 255) # clip values
        # ax[2, 1].imshow(color_add_B)
        # ax[2, 1].set_title("showing target mask (blue)")
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        if save == True:
            current_figure.savefig(result_filename, dpi=1500, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        """confusion matrix for binary classification"""
        # radar = truth, crestline = prediction
        #                   on radar     NOT on radar
        # on crestline      true pos     false pos
        # NOT on crestline  false neg    true neg
        true_positive = np.sum(radar_fatten * warped_fatten)
        false_positive = np.sum(warped_fatten) - true_positive
        false_negative = np.sum(radar_fatten) - true_positive
        true_negative = image_lines_mask.size - (true_positive + false_positive + false_negative)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(radar_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(warped_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(radar_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(warped_fatten)
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
        return radar_to_warped_score, warped_to_radar_score, result_filename, confusion_matrix
    def measure_close_points(self, radius=10, composite=False):
        """ input: object name, radar mask, crestlines mask,
            output: radar-to-warped score, warped-to-radar score, name of resulting image
            score from calling Close_Points in Metric"""
        print(f"Processor:measure_close_points")
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
        ax[0, 0].imshow(Image.open('image_with_crestlines.png'))
        ax[0, 0].set_title("warped crestlines over radar image")
        if composite==True:
            # plotter = Plot_2D(init_color='black', warp_color='red', thickness=self.thic)
            # f, ax = plt.subplots(figsize=(32, 32)) # NOT subplot()
            # ax.set_xlim(0, self.x_length)
            # ax.set_ylim(0, self.y_length)
            img = np.asarray((Image.open('need_ridges.png').convert('RGB')))
            x_length, y_length, _ = img.shape
            for x in range(x_length):
                for y in range(y_length):
                    if self.mask_radar_ridges[x][y] == 1:
                        img[x][y] = [255, 0, 0]
            ax[0, 1].imshow(img)
            # ax.autoscale()
            # current_figure = plt.gcf()
            # plt.show()
            # plt.draw()
            # plotter.Plot_SaveAxes(f, ax, current_figure, filename='image_with_crestlines.png') 
        else:
            ax[0, 1].imshow(self.radar_ridges_over_radar)
        ax[0, 1].set_title("image ridges over radar")
        # one direction
        print("radar to warped")
        radar_to_warped_score, radar_to_warped_hits, radar_fatten = Close_Points(self.mask_radar_ridges, self.mask_crestlines, radius)
        color_hits = np.stack((radar_to_warped_hits * 0, radar_to_warped_hits * 255, radar_to_warped_hits * 0), axis=2)
        didnt_hits = self.mask_radar_ridges - radar_to_warped_hits
        color_hits += np.stack((didnt_hits * 255, didnt_hits * 0, didnt_hits * 0), axis=2)
        color_hits = np.clip(color_hits, 0, 255) # clip values
        ax[1, 0].imshow(color_hits)
        ax[1, 0].set_title(f"hit (green) vs didn't hit (red): {radar_to_warped_score}")
        # add_B = self.mask_crestlines - self.mask_radar_ridges
        # color_add_B = color_hits + np.stack((add_B * 0, add_B * 0, add_B * 255), axis=2)
        # color_add_B = np.clip(color_add_B, 0, 255) # clip values
        # ax[1, 1].imshow(color_add_B)
        # ax[1, 1].set_title("showing target mask (blue)")
        # the other direction
        print(f"warped to radar")
        warped_to_radar_score, warped_to_radar_hits, warped_fatten = Close_Points(self.mask_crestlines, self.mask_radar_ridges, radius)
        color_hits = np.stack((warped_to_radar_hits * 0, warped_to_radar_hits * 255, warped_to_radar_hits * 0), axis=2)
        didnt_hits = self.mask_crestlines - warped_to_radar_hits
        color_hits += np.stack((didnt_hits * 255, didnt_hits * 0, didnt_hits * 0), axis=2)
        color_hits = np.clip(color_hits, 0, 255) # clip values
        ax[1, 1].imshow(color_hits)
        ax[1, 1].set_title(f"hit (green) vs didn't hit (red): {warped_to_radar_score}")
        # add_B = self.mask_radar_ridges - self.mask_crestlines
        # color_add_B = color_hits + np.stack((add_B * 0, add_B * 0, add_B * 255), axis=2)
        # color_add_B = np.clip(color_add_B, 0, 255) # clip values
        # ax[2, 1].imshow(color_add_B)
        # ax[2, 1].set_title("showing target mask (blue)")
        current_figure = plt.gcf()
        plt.show()
        plt.draw()
        result_name = f'{self.name}_result.png'
        current_figure.savefig(result_name, dpi=1000, pad_inches=0, facecolor=f.get_facecolor(), bbox_inches='tight')
        """confusion matrix for binary classification"""
        # radar = truth, crestline = prediction
        #                   on radar     NOT on radar
        # on crestline      true pos     false pos
        # NOT on crestline  false neg    true neg
        true_positive = np.sum(radar_fatten * warped_fatten)

        # save the true positives
        true_pos_radar_bmp_name = f'{self.name}_true_positive_radar_bitmap.txt'
        np.savetxt(fname=true_pos_radar_bmp_name, X=radar_to_warped_hits, fmt='%i')
        pos_truth_bmp_name = f'{self.name}_positive_truth_bitmap.txt'
        np.savetxt(fname=pos_truth_bmp_name, X=self.mask_radar_ridges, fmt='%i')
        
        false_positive = np.sum(warped_fatten) - true_positive
        false_negative = np.sum(radar_fatten) - true_positive
        true_negative = self.mask_radar_ridges.size - (true_positive + false_positive + false_negative)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(radar_fatten)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(warped_fatten)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(radar_fatten)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(warped_fatten)
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
        return radar_to_warped_score, warped_to_radar_score, result_name, confusion_matrix, true_pos_radar_bmp_name
    def Confusion_Matrix(self):
        """confusion matrix for binary classification"""
        # radar = truth, crestline = prediction
        #                   on radar     NOT on radar
        # on crestline      true pos     false pos
        # NOT on crestline  false neg    true neg
        mask_radar_ridges_neg = 1 - self.mask_radar_ridges
        mask_crestlines_neg   = 1 - self.mask_crestlines
        true_positive  = np.sum(self.mask_radar_ridges * self.mask_crestlines)
        false_positive = np.sum(self.mask_crestlines) - true_positive
        false_negative = np.sum(self.mask_radar_ridges) - true_positive
        true_negative  = np.sum(mask_radar_ridges_neg * mask_crestlines_neg)
        # error of omission: % false neg / all positive truths
        error_of_omission  = false_negative/np.sum(self.mask_radar_ridges)
        # error of comission: % false pos / all positive predictions
        error_of_comission = false_positive/np.sum(self.mask_crestlines)
        # user's accuracy: % true pos / all positive truths <- 1 - error of omission
        users_accuracy     = true_positive/np.sum(self.mask_radar_ridges)
        # producer's accuracy: % true pos / all positive predictions <- 1 - error of omission
        producers_accuracy = true_positive/np.sum(self.mask_crestlines)
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
        return confusion_matrix

"""modified version of displayContours from ridge-detection package"""
# original: https://pypi.org/project/ridge-detection/
RED_PIXEL_LINE = (255, 0, 0)
GREEN_PIXEL_CONTOUR = (0, 255, 0)
SIZE_RAY_JUNCTION = 1

from mrcfile import open as mrcfile_open
from copy import deepcopy
from PIL import ImageDraw
from math import exp, sin ,cos

def displayContours_mod(params, result, resultJunction):
    try:
        img=Image.fromarray(mrcfile_open(params.config_path_to_file).data).convert('RGB')
    except ValueError:
        img = Image.open(params.config_path_to_file).convert('RGB')
    pixelMap2 = img.load()
    sizePixelMap2 = img.size
    y, x, channels = np.asarray(img).shape
    binary_lines = np.zeros((y, x))
    """ plot the lines""" # lines are red
    if isinstance(result, list) is True:
        for line in result:
            for i, j in zip(line.col,line.row):
                pixelMap2[int(i), int(j)] = RED_PIXEL_LINE
                binary_lines[int(i)][int(j)] = 1
    img_only_lines = deepcopy(img)
    """ plot the contours""" # boundary of line are green
    if isinstance(result, list) is True:
        for cont in result:
            last_w_l ,last_w_r,px_r,px_l,py_r,py_l = 0,0,0,0,0,0
            for j in range(cont.num):
                px = cont.col[j]
                py = cont.row[j]
                nx = sin(cont.angle[j])
                ny = cos(cont.angle[j])
                if params.get_estimate_width():
                    px_r = px + cont.width_r[j] * nx
                    py_r = py + cont.width_r[j] * ny
                    px_l = px - cont.width_l[j] * nx
                    py_l = py - cont.width_l[j] * ny
                    if last_w_r > 0 and cont.width_r[j] > 0 and sizePixelMap2[0]>int(px_r)+1 and sizePixelMap2[1]>int(py_r)+1:
                        pixelMap2[int(px_r)+1, int(py_r)+1] = GREEN_PIXEL_CONTOUR
                    if last_w_l > 0 and cont.width_l[j] > 0 and sizePixelMap2[0]>int(px_l)+1 and sizePixelMap2[1]>int(py_l)+1:
                        pixelMap2[int(px_l) + 1, int(py_l) + 1] = GREEN_PIXEL_CONTOUR
                    last_w_r = cont.width_r[j]
                    last_w_l = cont.width_l[j]
    """ draw a circle (with ray SIZE_RAY_JUNCTION) centered in each junctions"""
    if params.get_show_junction_points() is True and isinstance(resultJunction, list) is True:
        for junction in resultJunction:
            draw = ImageDraw.Draw(img)
            draw.ellipse((int(junction.x) - SIZE_RAY_JUNCTION, 
                          int(junction.y) - SIZE_RAY_JUNCTION, 
                          int(junction.x) + SIZE_RAY_JUNCTION, 
                          int(junction.y) + SIZE_RAY_JUNCTION), 
                          fill = 'blue')
    if params.get_preview() is True:
        img.show()
    return img, img_only_lines, binary_lines