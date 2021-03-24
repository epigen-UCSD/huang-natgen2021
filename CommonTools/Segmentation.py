import unet_pytorch2 as unet
import cv2
import torch
import os
from torch.nn.parallel.data_parallel import DataParallel
import scipy.ndimage as ndi
import numpy as np
from tqdm import tqdm_notebook as tqdm

def get_model(model_fl):
    """Load standard neural network model"""
    model = unet.UNet()
    load_dic = torch.load(model_fl, map_location=torch.device('cpu'))
    try:
        model.load_state_dict({k[7:]:v for k,v in load_dic['state_dict'].items()})
    except:
        model.module.load_state_dict(load_dic['state_dict'])
    return model

def apply_model(im_,model,smx=128,smy=128,nsteps=100,cuda=False):
    if type(model) is str:
        model = get_model(model)
    sx,sy = im_.shape
    dx,dy = int(smx/2),int(smy/2)
    dmx,dmy = int(smx/4),int(smy/4)
    imf = []
    ims_4_vals = []
    for i in np.arange(sx/smx,dtype=int):
        for j in np.arange(sy/smy,dtype=int):
            im__0 = im_[i*smx:(i+1)*smx,j*smy:(j+1)*smy]
            im__1 = im_[dx+i*smx:dx+(i+1)*smx,j*smy:(j+1)*smy]
            im__2 = im_[i*smx:(i+1)*smx,dy+j*smy:dy+(j+1)*smy]
            im__3 = im_[dx+i*smx:dx+(i+1)*smx,dy+j*smy:dy+(j+1)*smy]
            im_val = []
            for im___ in [im__0,im__1,im__2,im__3]:
                smx_,smy_ = im___.shape
                if smx_==smx and smy_==smy:
                    imf.append(im___[np.newaxis,...].astype(np.float32))
                    im_val.append(True)
                else:
                    im_val.append(False)
            ims_4_vals.append(im_val)
    imf = np.array(imf)
    ntot = int(np.ceil(len(imf)/float(nsteps)))
    decompf=[]
    for i in range(ntot):
        decomp = imf[i*nsteps:(i+1)*nsteps]
        if cuda:
            decomp = model(torch.FloatTensor(decomp).cuda())
        else:
            decomp = model(torch.FloatTensor(decomp))
        decompf_ = torch.sigmoid(decomp).cpu().detach().numpy()
        decompf.extend(decompf_)
    decompf = np.array(decompf)
    im_decompf = np.zeros([sx,sy])
    count = 0
    countim = 0

    for i in np.arange(sx/smx,dtype=int):
        for j in np.arange(sy/smy,dtype=int):
            im0val,im1val,im2val,im3val = ims_4_vals[count]
            if im0val:
                im__0 = decompf[countim,0]
                countim+=1
                im_decompf[i*smx:(i+1)*smx,j*smy:(j+1)*smy] = im__0#[dmx:-dmx,dmy:-dmy]
            if im1val: countim+=1
            if im2val: countim+=1
            if im3val: countim+=1
            count+=1
    count = 0
    countim = 0
    for i in np.arange(sx/smx,dtype=int):
        for j in np.arange(sy/smy,dtype=int):
            im0val,im1val,im2val,im3val = ims_4_vals[count]
            if im0val:
                im__0 = decompf[countim,0]
                countim+=1
                im_decompf[dmx+i*smx:(i+1)*smx-dmx,dmy+j*smy:(j+1)*smy-dmy] = im__0[dmx:-dmx,dmy:-dmy]
            if im1val:
                im__1 = decompf[countim,0]
                countim+=1
                im_decompf[dmx+dx+i*smx:(i+1)*smx+dx-dmx,dmy+j*smy:(j+1)*smy-dmy] = im__1[dmx:-dmx,dmy:-dmy]
            if im2val:
                im__2 = decompf[countim,0]
                countim+=1
                im_decompf[dmx+i*smx:(i+1)*smx-dmx,dmy+dy+j*smy:(j+1)*smy+dy-dmy] = im__2[dmx:-dmx,dmy:-dmy]
            if im3val:
                im__3 = decompf[countim,0]
                countim+=1
                im_decompf[dmx+dx+i*smx:(i+1)*smx+dx-dmx,dmy+dy+j*smy:(j+1)*smy+dy-dmy] = im__3[dmx:-dmx,dmy:-dmy]
            count+=1
    return im_decompf
    
def get_contour(thresh):
    """
    Returns the contour (a list of coordinates Nx2) for a 2D binary image
    """
    try:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except: 
        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return np.squeeze(contours[0])
    
def solidity(thresh):
    """
    Calculate the solidity of a shape, which is the ratio of its area to the
    area of its convex hull.
    """
    contour = get_contour(thresh)
    hull = cv2.convexHull(contour)
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    return float(contour_area) / hull_area if hull_area >0 else 0
    
def segment_2d(mid_edge,min_area=300,max_area=3000,solidity_th=0.9,infocus_fraction_th=0):
        #mid_infocus = self.im_infocus[self.mid]
        mid_edge_ = (mid_edge>0).astype(np.uint8)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mid_edge_)

        mid_filter = np.zeros_like(labels)
        mid_centers,mid_stats = [],[]
        count=0
        #min_area,max_area,infocus_fraction_th,solidity_th = self.min_area,self.max_area,self.infocus_fraction_th,self.solidity_th
        for icell,stat in enumerate(stats):
            ymin,xmin,yw,xw,area = stat
            xmax,ymax = xmin+xw,ymin+yw
            #area cutting
            keep = (area<max_area)&(area>min_area)
            if not keep: continue
            #edge of image cutting
            keep = not (xmin==0 or ymin==0 or xmax==labels.shape[0] or ymax==labels.shape[1])
            if not keep: continue

            cell_im = labels[xmin:xmax,ymin:ymax]==icell
            #in focus cutting
            #infocus_fraction = np.sum(mid_infocus[xmin:xmax,ymin:ymax][cell_im])/float(area)
            #keep = infocus_fraction>infocus_fraction_th
            if not keep: continue
            #solidity cutting
            solidity_ = solidity(cell_im.astype(np.uint8))
            keep = solidity_>solidity_th
            if not keep: continue
            count+=1
            mid_filter[xmin:xmax,ymin:ymax][cell_im]=count
            mid_centers.append(centroids[icell])
            mid_stats.append([xmin,xmax,ymin,ymax,area])
        mid_centers = np.array(mid_centers)
        mid_stats = np.array(mid_stats)
        return mid_filter
    
def blurnorm2d(im,gb):
    """Normalize an input 2d image <im> by dividing by a cv2 gaussian filter of the image"""
    im_ = im.astype(np.float32)
    blurred = cv2.blur(im_,(gb,gb))
    return im_/blurred

def segment_fov(im, model):
    im_ = blurnorm2d(im,75)[::2,::2]
    return apply_model(im_,model,smx=128,smy=128,nsteps=100,cuda=False)
    
def filter_segmented_fov(imf, imf_thresh, min_area, max_area, solidity_th, infocus_fraction_th):
    return segment_2d(1-(imf>imf_thresh),min_area=min_area,max_area=max_area,solidity_th=solidity_th,infocus_fraction_th=infocus_fraction_th)
    
def run_segmentation(analysis_folder, ims, model, imf_thresh=0.075, min_area=1000, max_area=10000, solidity_th=0.9, infocus_fraction_th=0):
    im_labels = []
    for im_ in tqdm(ims):
        imf = segment_fov(im_, model)
        im_label = filter_segmented_fov(imf, imf_thresh=imf_thresh, min_area=min_area, max_area=max_area, solidity_th=solidity_th, infocus_fraction_th=infocus_fraction_th)
        im_labels.append(im_label)
    np.save(analysis_folder+os.sep+'DAPI_images_segmented.npy',np.array(im_labels,dtype=np.uint8))

def script_segmentation(analysis_folder):
    """Run the segmentation with default settings for non-interactive pipeline script"""
    model = get_model(r'C:\Users\ckern\edge_HCT15_last')
    ims = np.load(analysis_folder+os.sep+'DAPI_images.npy')
    run_segmentation(analysis_folder, ims, model)
    