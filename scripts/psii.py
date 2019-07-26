#! python
# run at cmd:
# scripts/psii.py -i1 "data\psII\A1-SILK_dark-2018-12-10 20_01_50-PSII0_1_0_0.tif" -i2 "data\psII\A1-SILK_dark-2018-12-10 20_01_50-PSII0_2_0_0.tif" -o debug -D print 
# -writeimg True -r results.json

import sys, traceback 
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv

### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i1", "--fmin", help="Input image file.", required=True)
    parser.add_argument("-i2", "--fmax", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.")
    parser.add_argument("-writeimg", "--writeimg")
    parser.add_argument("-r","--result", help="Result file.", required= True )
    args = parser.parse_args()
    return args

### Main workflow
def main():
    # Get options
    args = options()

    pcv.params.debug=args.debug #set debug mode
    pcv.params.debug_outdir=args.outdir #set output directory

    # Read image (converting fmax and track to 8 bit just to create a mask, use 16-bit for all the math)
    fmax, path, filename = pcv.readimage(args.fmax)

    # Threshold the image
    mask = pcv.threshold.binary(gray_img=fmax, threshold=20, max_value=255, 
                                   object_type='light')
    mask = pcv.fill(mask, 100)

    # Identify objects
    id_objects,obj_hierarchy = pcv.find_objects(img=fmax, mask=mask)

    # Define ROI
    roi1, roi_hierarchy = pcv.roi.rectangle(img=mask, x=180, y=90, h=200, w=200)

    # Decide which objects to keep
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=mask, roi_contour=roi1, 
                                                               roi_hierarchy=roi_hierarchy, 
                                                               object_contour=id_objects, 
                                                               obj_hierarchy=obj_hierarchy, 
                                                               roi_type='partial')
    # Object combine kept objects
    obj, masked = pcv.object_composition(img=mask, contours=roi_objects, hierarchy=hierarchy3)

    ################ Analysis ################  

    outfile=False
    if args.writeimg==True:
        outfile=args.outdir+"/"+filename

    # Find shape properties, output shape image (optional)
    shape_img = pcv.analyze_object(img=mask, obj=obj, mask=masked)

    # Fluorescence Measurement (read in 16-bit images)
    fmin = cv2.imread(args.fmin, -1)
    fmax = cv2.imread(args.fmax, -1)

    fvfm_images = pcv.fluor_fvfm(fdark=np.zeros_like(fmax), fmin=fmin, fmax=fmax, mask=kept_mask, bins=256)

    # Store the two images
    fv_img = fvfm_images[0]
    fvfm_hist = fvfm_images[1]
    fvfm = np.divide(fv_img, fmax, out=np.zeros_like(fmax, dtype='float'), where= np.logical_and(mask>1,fmax>0))

    # Pseudocolor the Fv/Fm grayscale image that is calculated inside the fluor_fvfm function
    pseudocolored_img = pcv.visualize.pseudocolor(gray_img=fvfm, mask=kept_mask, cmap='viridis',min_value=0, max_value=1)

    # Write shape and nir data to results file
    pcv.print_results(filename=args.result)

if __name__ == '__main__':
    main()