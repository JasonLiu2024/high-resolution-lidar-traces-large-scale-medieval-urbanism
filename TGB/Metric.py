import numpy as np
def Dice_Coefficient(image_1, image_2):
    """image_1 & image_2 are BOOL ARRAYs"""
    intersection = np.sum(image_1 * image_2)
    return intersection * 2.0 / (np.sum(image_1) + np.sum(image_2))

def Close_Points(examinER, examinEE, radius):
    """ DOES: for ea position, check if it's got neighbor in the same position in a different image; do so for each pixel of interest -> get percentage of pixels with neighbor 
        \nexaminER: (bool array) percentage of this image
        \nexaminEE: (bool array) the other image <- SAME shape as examinER
        \nradius: (int, int) range in which a neighbor COUNTs 
        RETURNS: 
        \n score: % hits
        \n bmp for pixels that hit
        \n fatten: fattened ver of examinER used"""
    y, x = examinER.shape # x, y are flipped to fit math conventions
    total_interest = np.sum(examinER)
    hits = 0
    hits_display = np.zeros(shape=(y, x))
    fatten = np.zeros(shape=(y, x))
    for row in range(y):
        for col in range(x):
            if(examinER[row][col] == 1):
                d = max(row - radius, 0)
                u = min(row + radius + 1, y)
                l = max(col - radius, 0)
                r = min(col + radius + 1, x)
                # print(f"checking: {row, col}")
                # print(f"{examinER[d:u, l:r]}")
                # print("vs")
                # print(f"{examinEE[d:u, l:r]}")
                find = np.sum(examinEE[d:u, l:r])
                fatten[d:u, l:r] = 1
                if(find >= 1):
                    hits += 1
                    hits_display[row][col] = 1
                    # print("HIT")
    score = hits / float(total_interest)
    print(f"% hits {score}, from {hits}/{int(total_interest)}")
    return score, hits_display.astype(int), fatten.astype(int)

def Kappa_Binary(true_positive : int, true_negative : int, false_positive : int, false_negative : int):
    """Cohen's Kappa (or Kappa's coefficient) for binary classification"""
    print(f"<<<<<<Kappa calculation>>>>>>")
    everything = float(true_positive + true_negative + false_positive + false_negative)
    print(f"\tEverything: {everything}")
    Po = (true_positive + true_negative)/everything
    print(f"Po: (true positive + true_negative) / everything: {Po}")
    # print(f"Po: {Po}")
    Pyes = ((true_positive + false_positive) * 
            (true_positive + false_negative))/everything ** 2
    print(f"Pyes: {Pyes}")
    Pno = ((true_negative + false_positive) * 
            (true_negative + false_negative))/everything ** 2
    print(f"Pno: {Pno}")
    Pe = Pyes + Pno
    print(f"Pe: {Pe}")
    kappa = (Po - Pe)/(1 - Pe)
    return kappa