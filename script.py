import argparse 
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

SEED = 42           # Seed for noise generation
SHOWME = False      # If SHOWME == True the cose show the images

# Degrading the image by decreasing the contrast and by adding noise 
def degrade(img, rng):
    low = cv2.convertScaleAbs(img, alpha = 0.6, beta = 10)                             # Lower the contrast
    noisy = low.astype(np.float32) + rng.normal(0, 5, img.shape)       # Addition of the noise
    return np.clip(noisy, 0, 255).astype(np.uint8)                      # Return the image with the correct dimension

# Linear contrast stretching
def stretch_v(img, p_lo = 1, p_hi = 99):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                      # HSV conversion
    v = hsv[..., 2].astype(np.float32)                                              # Extraction of the V-channel
    lo, hi = np.percentile(v, (p_lo, p_hi))                                         # Defining the new "min" and "max": valus below lo become black while the ones above hi become white
    hsv[..., 2] = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)      # Linear contrast stretching formula
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)                                     # Changing back the image to BGR

# Histogram equalization
def equalize_v(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          # HSV conversion
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])         # Histogram equalization function
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)         # Change back the image to BGR                       
    
# Non local means combined with the linear contrast stretching
def nlm_then_stretch(img):
    den = cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)       # Nlm Function
    return stretch_v(den)                                               # Applying the linear contrast stretching function to the filtered image

# Evaluation of the results through PSNR, SSIM and MAE
def metrics(ref, out):
    return (cv2.PSNR(ref, out), ssim(ref, out, channel_axis=2, data_range=255), float(np.mean(np.abs(ref.astype(np.float32) - out))))

# Dictionary of the functions
METHODS = {"degraded": lambda x: x, "stretch": stretch_v, "equalize": equalize_v, "nlm_stretch": nlm_then_stretch}

def main():
    # Creation of the parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="images")                        # Original images folder
    ap.add_argument("--output", default="results")                      # Output folder with processed images and the metric.csv file
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []

    # Main cycle - Extraction of the images from the folder
    for p in sorted(Path(args.input).glob("*.*")):                                      # Sorting the images --> rng is shared, having a fixed order is useful to reproduce the noise consistently
        ref = cv2.imread(str(p))                                                        # Original image
        
        if ref is None:
            continue
        
        img_dir = out / p.stem                                                          # Creating subfolders for each original image in the output folder
        img_dir.mkdir(parents=True, exist_ok=True)

        deg = degrade(ref, rng)                                                         # Creation of the degraded image
        
        # Application of each process to the image
        for name, fn in METHODS.items():
            res = fn(deg)
            cv2.imwrite(str(img_dir / f"{name}.png"), res)                              # Saving the resulting image in the output folder
            psnr, s, mae = metrics(ref, res)                                            # Evaluating the process
            rows.append([p.stem, name, psnr, s, mae])

            # Titles for the printed images
            if name == "degraded":
                left, title = ref, f"Original - {name}"
            else:
                left, title = deg, f"Degraded - {name}"

            if SHOWME == True:
                cv2.imshow(title, np.hstack([left, res]))                                   # Showing the comparison between the images
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    # Creation of the metrics table
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "method", "PSNR", "SSIM", "MAE"])
        w.writerows(rows)

if __name__ == "__main__":
    main()