#External packages
import sys,glob,os
import numpy as np
import pickle
import matplotlib.pylab as plt
from tqdm import tqdm_notebook as tqdm
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
#from tqdm import tqdm

#imports
import IOTools as io
import FittingTools as ft
import AlignmentTools as at
import Segmentation as seg
#for exploring the data
import MaxViewer as mv

def load_data(master_folders, analysis_folder):
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    #get a list of hybe folders
    folders = ft.flatten([[folder for folder in glob.glob(master_folder+os.sep+'*') 
               if os.path.isdir(folder) and os.path.basename(folder)[0]=='H' and 
               np.any([let in os.path.basename(folder) for let in ['R']]) and
               np.all([let not in os.path.basename(folder)for let in ['H0','dapi']])
              ] for master_folder in master_folders])
    #keep complete sets
    num_daxs = np.array([len(glob.glob(folder+os.sep+'*.dax')) for folder in folders])
    folders = np.array(folders)[num_daxs==np.max(num_daxs)]
    #sort them by the order they were hybed
    folders = np.array(folders)[np.argsort(list(map(io.hybe_number,folders)))]#[2:]
    print("Found "+str(len(folders))+' folders.')
    fovs = np.sort(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))))

    h0_folder = glob.glob(master_folders[0]+os.sep+'H0*')[0]
    #Outputs: folders,fovs
    return folders, fovs, h0_folder
    
def load_dapi_dax_frame(h0_folder, fovs, frame):
    return [io.DaxReader(h0_folder+os.sep+fov).loadAFrame(frame*5-1) for fov in fovs]#10*5-1

def save_dapi_images(analysis_folder, ims):
    #Save DAPI images for segmentation
    np.save(analysis_folder+os.sep+'DAPI_images.npy',ims)
    
def save_median_image_across_fovs(folders, analysis_folder):
    return ft.get_ims_meds(folders[2],num_cols=4,fovs=None,
                           median_fl=analysis_folder+os.sep+'im_repairs_3col.npy',
                           overwrite=True,tag='start',verbose=True,func=np.median)
                              
def flat_field_correction(folders, analysis_folder, fovs, force_rerun=False):
    im_repairs = np.load(analysis_folder+os.sep+'im_repairs_3col.npy')

    # don't need all rounds of hybridization to compute drift

    # we only need a portion of the rounds f hybridization because we just want to find average position
    # across most of the rounds. As such, we only need to correct drift in the ones we are using.
    folders_keep = folders[:]  # chose 10 rounds when there was not a large drift.
    for fov in tqdm(fovs):
        print(fov)
        col_tags = ['750','647','561','beads']
        ncolors_signal = len(col_tags)-1
        #decide where to save data:
        dax_fls = [analysis_folder+os.sep+fov.replace('.dax','_mean_'+str(icol)+'.dax') 
                   for icol in range(ncolors_signal)]
        
        #finalfit_fl = dax_fl.replace('.dax','_finalfit.npy')
        
        if not np.all([os.path.exists(fl) for fl in dax_fls]) or force_rerun:#True
            # map data into image objects of type dax_im. See IOTools for documentaion of dax_im class##############
            
            daxs_signal,names_signal,daxs_beads,names_beads = io.get_ims_fov_daxmap(folders_keep,fov,
                                                                                         col_tags = col_tags,
                                                                                    pad = 0)
            
            #compute pixel drift ######
            drift_fl = dax_fls[0].replace('.dax','_drift.npy')
            if os.path.exists(drift_fl) and not force_rerun: tzxys = np.load(drift_fl)
            else:
                sz=400 # this is the size of the square of the beads image used for registration
                dax_bead_ref = daxs_beads[0] # the image object use for registration
                # select two non-overlapping squares of the beads field of view to check for consistency of drift.
                im_ref1 = ft.grab_block(dax_bead_ref,np.array(dax_bead_ref.shape)/2,[sz]*3)
                im_ref2 = ft.grab_block(dax_bead_ref,np.array(dax_bead_ref.shape)/2+[0,sz,0],[sz]*3)
                tzxys = []
                for dax_bead in daxs_beads:
                    im1 = ft.grab_block(dax_bead,np.array(dax_bead.shape)/2,[sz]*3)
                    im2 = ft.grab_block(dax_bead,np.array(dax_bead.shape)/2+[0,sz,0],[sz]*3)

                    tzxy1 = ft.fft3d_from2d(im_ref1,im1)
                    tzxy2 = ft.fft3d_from2d(im_ref2,im2)
                    tzxys.append([tzxy1,tzxy2])

                    #print([tzxy1,tzxy2])
                np.save(drift_fl,tzxys)
            #compute and save mean###########
            inds = range(len(daxs_signal))
            
            #print("Loading and averaging per color...")
            
            for icol,dax_fl in enumerate(dax_fls):
                im_mean = np.zeros(daxs_signal[0].shape,dtype=np.float32)
                nim = 0
                print(dax_fl)
                if not os.path.exists(dax_fl) or force_rerun:
                    for idax in tqdm(inds):
                        dax = daxs_signal[idax]
                        ibead = int(idax/(dax.num_col-1))
                        tzxy1,tzxy2 = tzxys[ibead]
                        if idax%ncolors_signal==icol: #this keeps only the 750 color
                            if np.max(np.abs(tzxy1-tzxy2))<=1: #keep if drift error is within a pixel
                                im = dax.get_im().astype(float)
                                im = im/im_repairs[dax.color]
                                im = ft.translate(im,-tzxy1[[0,2,1]])#a mistranspose here for some reason
                                zlim = np.min([im_mean.shape[0],im.shape[0]])
                                im_mean[:zlim]+=im[:zlim]
                                nim+=1
                        #dax_fl = analysis_folder+os.sep+fov.replace('.dax','_mean'+str(imn)+'.dax')
                        #io.writeDax(im_means[imn],dax_fl)
                    im_mean_save = im_mean/float(nim)
                    np.save(dax_fl.replace('.dax','_min-max.npy'),[np.min(im_mean_save),np.max(im_mean_save)])
                    io.writeDax(im_mean_save,dax_fl)
                    
def get_candidate_spots(analysis_folder, colors):
    daxs = [glob.glob(analysis_folder+os.sep+f'*_mean_{color}.dax') for color in colors]
    #daxs0 = glob.glob(analysis_folder+os.sep+f'*_mean_{colors[0]}.dax')
    #daxs1 = glob.glob(analysis_folder+os.sep+f'*_mean_{colors[1]}.dax')
    ims_segmented = np.load(analysis_folder+os.sep+'DAPI_images_segmented.npy')
    isegm = 0
    #for dax_fl0,dax_fl1 in tqdm(list(zip(daxs0, daxs1))):
    for dax_round in tqdm(list(zip(*daxs))):
        
        ##load average image for color set 1 (750nm)
        #im_mn = io.DaxReader(dax_fl0).loadAll()
        #min_,max_ = np.load(dax_fl0.replace('.dax','_min-max.npy'))
        #im_mn0 = 1.*im_mn/np.max(im_mn)*max_+min_
        ##load average image for color set 2 (647nm)
        #im_mn = io.DaxReader(dax_fl1).loadAll()
        #min_,max_ = np.load(dax_fl1.replace('.dax','_min-max.npy'))
        #im_mn1 = 1.*im_mn/np.max(im_mn)*max_+min_
        
        zxyhs = []
        im_mns = []
        for dax in dax_round:
            im_all = io.DaxReader(dax).loadAll()
            min_,max_ = np.load(dax.replace('.dax', '_min-max.npy'))
            im_mn = 1.*im_all/np.max(im_all)*max_+min_
            im_mns.append(im_mn)
            zxyhs.append(ft.return_candidate_centers(im_mn,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
                                      plt_val=False,save_file=None))
        
        # fit the two images. ex,ey,ez are the distances to the edge of the image which are excluded.
        # gmax is the number of pixels to the gaussian correction to "flatten" the image
        #filtsz is local maximum within this many pixels
        #ths is brightness freshold (how many standard deviations above the camera/intrinsic noise of the staining)
        #zxyh1 = ft.return_candidate_centers(im_mn1,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
        #                              plt_val=False,save_file=None)
        
        #zxyh0 = ft.return_candidate_centers(im_mn0,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
        #                                  plt_val=False,save_file=None)
        #resize the segmented file. this was passed to the neural network at half the size
        im_segm = cv2.resize(ims_segmented[isegm],tuple(im_mn.shape[1:]),interpolation  = cv2.INTER_NEAREST)
        
        #### get the two brightest spots per cell in the combined fits per 2 colors  ########
        #zxyh0_keep = ft.get_2brighspots_per_cell(im_segm,zxyh0,dist = 10)
        #zxyh1_keep = ft.get_2brighspots_per_cell(im_segm,zxyh1,dist = 10)
        
        ############# COLIN, PLEASE NOTE THAT I AM RETURNING THE CELL IDS HERE NOW!!! ###############
        #zxyh_keep, cell_ids = ft.get_2brighspots_per_cell(im_segm, isegm, np.concatenate([zxyh0,zxyh1],axis=-1),dist = 20)
        zxyh_keep, cell_ids = ft.get_2brighspots_per_cell(im_segm, isegm, np.concatenate(zxyhs,axis=-1),dist = 20)
        
        isegm+=1
        
        np.save(dax_round[0].replace('.dax','_fits-per-cell.npy'),zxyh_keep)
        np.save(dax_round[0].replace(f'mean_{colors[0]}.dax','_cell_ids.npy'),cell_ids)
        
        #### save image for inspection   ##############
        while len(im_mns) < 3:
            im_mns.append(im_mns[0])
        imf=np.dstack([at.minmax(ft.blurnorm2d(np.max(im_mn,axis=0),30),0,2) for im_mn in im_mns])
                
        fig = plt.figure(figsize=(40,40))
        plt.imshow(imf,cmap='gray',vmax=2)
        z,x,y,h = zxyh_keep
        plt.plot(y,x,'o',markersize=12,markeredgewidth=1,markeredgecolor='b',markerfacecolor='None')
        plt.contour(im_segm,[0.5])
        fig.savefig(dax_round[0].replace('.dax','_spots_selected_final.png'))
        plt.close()
        
    fit_files = glob.glob(analysis_folder+os.sep+'*_fits-per-cell.npy')
    cell_id_fls = glob.glob(analysis_folder+os.sep+'*_cell_ids.npy')
    coords=[np.load(fl).T for fl in fit_files]
    def fl_to_fov(fl): return int(os.path.basename(fl).split('_mean')[0].split('_')[-1])
    dic_save = {}
    dic_save['class_ids'] = ft.flatten([[fl_to_fov(fl)]*len(coord) for coord,fl in zip(coords,fit_files)])
    #print(coords[0])
    dic_save['coords'] = ft.flatten([coord[:,[1,2,0]] for coord in coords])
    dic_save['names']= [os.path.basename(fl).split('_mean')[0] for fl in fit_files]
    dic_save['cell_ids'] = ft.flatten([ np.load(cell_ids) for cell_ids in cell_id_fls])
    pickle.dump(dic_save,open(analysis_folder+os.sep+'Selected_Spot.pkl','wb'))
    
def get_candidate_spots_new(analysis_folder, colors):
    daxs = []
    for color in colors:
        daxs.extend(glob.glob(analysis_folder+os.sep+f'*_mean_{color}.dax'))
    ims_segmented = np.load(analysis_folder+os.sep+'DAPI_images_segmented.npy')
    isegm = 0
    
    for ims in tqdm(ims_segmented):#dax_round in tqdm(list(zip(*daxs))):
        zxyhs = []
        im_mns = []
        for dax in dax_round:
            im_all = io.DaxReader(dax).loadAll()
            min_,max_ = np.load(dax.replace('.dax', '_min-max.npy'))
            im_mn = 1.*im_all/np.max(im_all)*max_+min_
            im_mns.append(im_mn)
            zxyhs.append(ft.return_candidate_centers(im_mn,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
                                      plt_val=False,save_file=None))
        
        # fit the two images. ex,ey,ez are the distances to the edge of the image which are excluded.
        # gmax is the number of pixels to the gaussian correction to "flatten" the image
        #filtsz is local maximum within this many pixels
        #ths is brightness freshold (how many standard deviations above the camera/intrinsic noise of the staining)
        #zxyh1 = ft.return_candidate_centers(im_mn1,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
        #                              plt_val=False,save_file=None)
        
        #zxyh0 = ft.return_candidate_centers(im_mn0,ex=20,ey=20,ez=2,gmax=10,filtsz=5,ths=4,
        #                                  plt_val=False,save_file=None)
        #resize the segmented file. this was passed to the neural network at half the size
        im_segm = cv2.resize(ims_segmented[isegm],tuple(im_mn.shape[1:]),interpolation  = cv2.INTER_NEAREST)
        
        #### get the two brightest spots per cell in the combined fits per 2 colors  ########
        #zxyh0_keep = ft.get_2brighspots_per_cell(im_segm,zxyh0,dist = 10)
        #zxyh1_keep = ft.get_2brighspots_per_cell(im_segm,zxyh1,dist = 10)
        
        ############# COLIN, PLEASE NOTE THAT I AM RETURNING THE CELL IDS HERE NOW!!! ###############
        #zxyh_keep, cell_ids = ft.get_2brighspots_per_cell(im_segm, isegm, np.concatenate([zxyh0,zxyh1],axis=-1),dist = 20)
        zxyh_keep, cell_ids = ft.get_2brighspots_per_cell(im_segm, isegm, np.concatenate(zxyhs,axis=-1),dist = 20)
        
        isegm+=1
        
        np.save(dax_round[0].replace('.dax','_fits-per-cell.npy'),zxyh_keep)
        np.save(dax_round[0].replace('mean_0.dax','_cell_ids.npy'),cell_ids)
        
        #### save image for inspection   ##############
        imf=np.dstack([at.minmax(ft.blurnorm2d(np.max(im_mn,axis=0),30),0,2) for im_mn in im_mns])
        
        fig = plt.figure(figsize=(40,40))
        plt.imshow(imf,cmap='gray',vmax=2)
        z,x,y,h = zxyh_keep
        plt.plot(y,x,'o',markersize=12,markeredgewidth=1,markeredgecolor='b',markerfacecolor='None')
        plt.contour(im_segm,[0.5])
        fig.savefig(dax_round[0].replace('.dax','_spots_selected_final.png'))
        plt.close()
        
    fit_files = glob.glob(analysis_folder+os.sep+'*_fits-per-cell.npy')
    cell_id_fls = glob.glob(analysis_folder+os.sep+'*_cell_ids.npy')
    coords=[np.load(fl).T for fl in fit_files]
    def fl_to_fov(fl): return int(os.path.basename(fl).split('_mean')[0].split('_')[-1])
    dic_save = {}
    dic_save['class_ids'] = ft.flatten([[fl_to_fov(fl)]*len(coord) for coord,fl in zip(coords,fit_files)])
    dic_save['coords'] = ft.flatten([coord[:,[1,2,0]] for coord in coords])
    dic_save['names']= [os.path.basename(fl).split('_mean')[0] for fl in fit_files]
    dic_save['cell_ids'] = ft.flatten([ np.load(cell_ids) for cell_ids in cell_id_fls])
    pickle.dump(dic_save,open(analysis_folder+os.sep+'Selected_Spot.pkl','wb'))
    
def drift_correction(fov_id, folders, analysis_folder):
    #flat field correction
    im_repairs = np.load(analysis_folder+os.sep+'im_repairs_3col.npy')

    #bleed through correction ignored
    #bleed_file = r'\\169.254.192.20\Chromatin_NAS_1\Bogdan\HCT_Alignment_multicolor\BBResult\bleed_mats.pkl'
    dic_bleed_slopes={}
    #dic_bleed_slopes = pickle.load(open(bleed_file,'rb'))

    #keep complete sets
    num_daxs = np.array([len(glob.glob(folder+os.sep+'*.dax')) for folder in folders])
    folders = np.array(folders)[num_daxs==np.max(num_daxs)]
    #sort them by the order they were hybed
    folders = np.array(folders)[np.argsort(list(map(io.hybe_number,folders)))]
    #print("Found "+str(len(folders))+' folders.')
    fovs = np.sort(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))))
    ##Load the selected chromosme data
    dic = pickle.load(open(analysis_folder+os.sep+'Selected_Spot.pkl','rb'))
    coords = np.array(list(dic['coords']))[:,[0,1,2]]
    chr_pts = ft.partition_map(coords,dic['class_ids'])

    fov_ids = np.unique(dic['class_ids'])

    ###paramaters
    overwrite=True
    force_drift=False
    verbose=False
    #drift paramaters
    ref=0 #drift reference hybe
    sz_ex=300 #the effective size of the cube to do drift correction
    hseed_beads=0 # The minimum bead height (prefitting)
    nbeads = 500 # The maximum number of beads

    #distance/numer of points cutoffs
    cutoff_window = 30 #fitted points within this distance will be reported
    cutoff_chr = 30 #if no prefered brightness points are detected within this distance, then include low brightness ones
    min_pts = 10 # number of fits/ window
    th_fluctuations = 2 # 2 std away for selecting a spot

    ##RUN
    print(fov_ids[fov_id])

    from scipy.spatial.distance import cdist

    #iterate through good fields of view for which chromosomes have been selected
    folders_keep = list(folders[:])
    #itterate through fields of view
    #Get selected chromosomal positions
    #for fov_id in tqdm(fov_ids):
    file_=fovs[fov_id]
    chr_pts_ = chr_pts[fov_id] #positions of the selected centers of chromosomes in reference frame.
    chr_pts_ = np.array(chr_pts_)[:,::-1]#zxy coords of chromosomes already in the right position

    #Decide where to save the candidate positions of the hybe
    fl_cands = analysis_folder+os.sep+file_.replace('.dax','__current_cand.pkl')#file where to save candidates
    fl_cor = analysis_folder+os.sep+file_.replace('.dax','__drift.pkl')#file where to save drift correction
    fl_cor_fls = analysis_folder+os.sep+file_.replace('.dax','__driftfiles.npy')

    #print(fl_cands)

    candid_spot={}

    #load data (pseudo-memory mapping)
    daxs_signal,names_signal,daxs_beads,names_beads = io.get_ims_fov_daxmap(folders_keep,file_,
                                                                         col_tags = None, pad = 0)
    #compute the drift for the field of view
    if len(daxs_beads)>1:
        txyz_both,_ = ft.get_STD_beaddrift_v2(daxs_beads,sz_ex=sz_ex,hseed=hseed_beads,nseed=nbeads,
                                           ref=None,force=force_drift,save_file=fl_cor,verbose=verbose)
        txyz = np.mean(txyz_both,1)
        txyz = np.array(txyz)-[txyz[ref]]
        np.save(fl_cor_fls,np.array(folders_keep))
    #repeat for colors
    #num_col = int(len(daxs_signal)/len(daxs_beads))
    #iterate through folders
    for iim in np.arange(len(daxs_signal))[:]:
        #iim specific vars
        im = daxs_signal[iim]

        ibead = im.index_beads
        icol = im.color
        tag = names_signal[iim]
        if verbose: print(tag)
        txyz_ = txyz[ibead]

        #fit and update dictionary
        #check if recalculated
        candid_dic,candid_spot,drift_dic = {},{},{}
        bad_drift = False
        if os.path.exists(fl_cands):
            candid_spot,drift_dic = pickle.load(open(fl_cands,'rb'))
            txyz_old = drift_dic.get(tag,np.array([np.inf,np.inf,np.inf]))
            if np.linalg.norm(txyz_old-txyz_)>0.5:
                bad_drift=True
            key = np.max(list(candid_spot.keys()))
            candid_dic,_ = candid_spot[key]
        ims_bleed = [dax_sig for dax_sig in daxs_signal if ibead==dax_sig.index_beads]
        color_names_bleed = [names_signal[idax_sig].split('_')[-1] for idax_sig,dax_sig in enumerate(daxs_signal) if ibead==dax_sig.index_beads]
        color_name = tag.split('_')[-1]
        dic_bleed  = {}
        dic_bleed['ims'] = ims_bleed
        dic_bleed['m_slopes'] = [dic_bleed_slopes.get('m_'+colnm_b+'->'+color_name,None) for colnm_b in color_names_bleed]

        if tag not in candid_dic or overwrite or bad_drift:
            ft.update_candid_spot(im,chr_pts_,txyz_,tag,drift_dic=drift_dic,dic_bleed=dic_bleed,th_std=th_fluctuations,num_pts=min_pts,
                            cutoff_window=cutoff_window,cutoff_chr=cutoff_chr,candid_spot=candid_spot,fl_cands=fl_cands,
                                  im_repair=im_repairs[icol],plt_val=False,verbose=verbose)
    
def drift_correction_batch(folders, analysis_folder, jobs=8):
    import multiprocessing
    import functools
    dic = pickle.load(open(analysis_folder+os.sep+'Selected_Spot.pkl','rb'))
    fov_ids = np.unique(dic['class_ids'])
    print(fov_ids)
    drift_partial = functools.partial(drift_correction, folders=folders, analysis_folder=analysis_folder)
    with multiprocessing.Pool(jobs) as pool:
        pool.map(drift_partial, fov_ids)
        
def em_algorithm(analysis_folder, chromatic_correction=None, use_center=True):
    ## Read files ############
    files = glob.glob(analysis_folder+os.sep+'*__current_cand.pkl')
    #files = [files[fov_num]]
    fovs = [os.path.basename(fl).split('__')[0]+'.dax'for fl in files]

    dic_spots,dic_drifts, dic_cell_ids = [],[],[]
    print("Loading candidates")
    for cand_fl in tqdm(files):
        cell_ids_fl = cand_fl.replace('__current_cand.pkl', '__cell_ids.npy')
        cell_ids = np.load(cell_ids_fl)
        dic_cell_ids.append(cell_ids)
        dic_spot,dic_drift = pickle.load(open(cand_fl,'rb'),encoding='latin1')#remove latin1 for py3 (maybe?)
        dic_spots.append(dic_spot)
        dic_drifts.append(dic_drift)
        
    fovs_spots = ft.flatten([[fov]*len(dic)for fov,dic in zip(fovs,dic_spots)])
    
    
    # this is being saved for later use in the QC steps. It is computing the fov assignments for each spot.
    # takes about 10 seconds to run.
    fov_ids = list()
    for idx,fov in enumerate(dic_spots):
        for i in fov:
            fov_ids.append(idx)
            
    fov_ids = np.array(fov_ids)
    saveloc = analysis_folder+os.sep+r"fov_ids_em_alg"
    np.save(saveloc, fov_ids)

    ###Final outputs: dic_spots,dic_drifts,fovs_spots(keep track for each chromosome which field of view it belongs to)
    # dic_spots - list of dictionaries for each analyzed field of view (fov)
    # each dic_spot is labelled by the index of the chromosome
    # (cand_dic,pos_chr) = dic_spots[fovi][chri] is the dictionary for fov fovi and for chri 
    # pos_chr is the z,x,y positin of the chromosomal location selected in the mean image across ~10 rounds of hybridization
    # cand_dic gives for each round of hyb and color it gives an array of fits.
    # Each fit consists of brightness, z,x,y position, background brightness, z,x,y widths of the gaussians
    collected_keys = []
    for dic_spots_fov in dic_spots[:1]:
        for chri in dic_spots_fov:
            (cand_dic,pos_chr) = dic_spots_fov[chri]
            collected_keys.extend(cand_dic.keys())
    collected_keys = np.unique(collected_keys)
    
    cols = ['750','647','561']
    regi_to_htag = {}
    for htag in collected_keys:
        col = htag.split('_')[-1]
        regnms = htag.split('_')[0].replace('B','R').replace(',','').split('R')[1:]
        coli = cols.index(col)
        regi_to_htag[regnms[coli]]=htag
    mod_range = range(42)
    htags = [regi_to_htag[str(ireg+1)] for ireg in mod_range]
    
    #Get chromatin correction
    if chromatic_correction:
        dic_chr = pickle.load(open(chromatic_correction,'rb'), encoding='latin1')
        zxy_647,zxy_750 = dic_chr['zxy_647_750']
        dic_chr['m_647_750'] = ft.calc_color_matrix(zxy_647,zxy_750)
        zxy_647,zxy_561 = dic_chr['zxy_647_561']
        dic_chr['m_647_561'] = ft.calc_color_matrix(zxy_647,zxy_561)
    else:
        dic_chr = {}
    

    ### Apply color correction
    dic_spots_cor = []
    tags = htags
    print("Applying color correction")
    for dic_spot in tqdm(dic_spots):
        dic_spot_ = {kspot:(ft.correct_color(dic_spot[kspot][0],tags,
                                             zxy_pix_size = [200,109,109],
                                            dic_chr=dic_chr,dic_bleed={}), # change dic_chr to apply or not
                                            dic_spot[kspot][1]) 
                                             for kspot in dic_spot}
        dic_spots_cor.append(dic_spot_)
            
    ### Initialize ########
    zxys_f,hs_f,cols_f  = [],[],[]

    print("Initializing")
    for dic_spot in tqdm(dic_spots_cor):
        for isp in dic_spot.keys():
            dic_cand = dic_spot[isp][0]
            zxys_T,hs_T,cols_T = ft.get_maxh_estimate(dic_cand,htags)
            zxys_f.append(zxys_T)
            hs_f.append(hs_T)
            cols_f.append(cols_T)
            
    ### Run to converge #########
    def refine_set(dic_spots,zxys_f,hs_f, use_local=True,use_center=use_center):
        #print("Lengths dics,zxys:",np.sum(list(map(len,dic_spots))),len(zxys_f))
        Dhs,Ddists,DdistsC = ft.get_Ddists_Dhs(zxys_f,hs_f,nint=5)
        zxys_f2,hs_f2,cols_f2,scores_f2,all_scores_f2  = [],[],[],[],[]
        i_ = 0
        # go through each individual spot in the zxys_f matrix of coordinates and get a statistical estimate 
        # for each candidate in each round. Maintain ordering. This simply refines the set of accepted points 
        # for each round for each spot in each FOV.
        for dic_spot in tqdm(dic_spots): # for each FOV
            for isp in dic_spot.keys(): # for each spot
                dic_cand = dic_spot[isp][0]
                zxys_T,hs_T,cols_T,scores_T,all_scores = ft.get_statistical_estimate(dic_cand,htags,Dhs,Ddists,DdistsC,
                                         zxys_T=zxys_f[i_],nint=5,use_local=use_local,use_center=use_center) # for each round
                zxys_f2.append(zxys_T)
                hs_f2.append(hs_T)
                cols_f2.append(cols_T)
                scores_f2.append(scores_T)
                all_scores_f2.append(all_scores)
                i_+=1
        return zxys_f2,hs_f2,cols_f2,scores_f2,all_scores_f2

    saved_zxys_f=[zxys_f]
    save_hs_f=[hs_f]
    #niter = 6
    same_fraction = 0
    num_ref=0
    use_local = True
    converged = 0
    while converged < 3:
    #for num_ref in range(niter):
        #use_local = num_ref<niter/2
        print('----------------')
        print('EM iteration number: ',num_ref+1)
        
        zxys_f,hs_f,cols_f,scores_f,all_scores_f = refine_set(dic_spots_cor,zxys_f,hs_f,use_local=use_local,use_center=False)
        saved_zxys_f.append(zxys_f)
        save_hs_f.append(hs_f)
        
        #check convergence
        dif = np.array(saved_zxys_f[-1])-np.array(saved_zxys_f[-2])
        nan =  np.all(np.isnan(dif),axis=-1)
        same = nan|np.all(dif==0,axis=-1)
        same_fraction = np.sum(same)/float(np.prod(same.shape))
        if same_fraction >= 0.99:
            converged += 1
        else:
            converged = 0
        print("fraction the same:",same_fraction)
        print("fraction nan:",np.sum(nan)/float(np.prod(nan.shape)))
        num_ref += 1
        
    return zxys_f, hs_f, cols_f, scores_f, all_scores_f, dic_spots, dic_drifts, dic_cell_ids, fovs_spots
    
def filter_chromosomes(num_segs, max_missing, zxys_f, hs_f, cols_f, scores_f, all_scores_f, dic_cell_ids, fovs_spots, cutoff_exp):
    # set to nan any below the threshold in a given round.
    tonan = np.array(scores_f)>np.exp(cutoff_exp)
    zxys_clean = np.array(zxys_f)
    zxys_clean[tonan==False]=np.nan

    keep = np.sum(np.isnan(zxys_clean[:,:num_segs,0]),axis=-1)<=max_missing # number of rounds it can be absent in
    zxys_clean_filtered = zxys_clean[keep]
    cell_ids_clean = np.array(ft.flatten(dic_cell_ids))[keep]
    hs_clean = np.array(hs_f)[keep]
    scores_clean = np.array(scores_f)[keep]
    fovs_spots_keep= np.array(fovs_spots)[keep]
    
    return keep, zxys_clean_filtered, cell_ids_clean, hs_clean, scores_clean, fovs_spots_keep
    
def make_post_analysis_folder(analysis_folder):
    post_analysis_folder = analysis_folder+os.sep+'Post_analysis'+os.sep
    if not os.path.exists(post_analysis_folder):
        os.makedirs(post_analysis_folder)
    return post_analysis_folder
    
# performs a mean correction for 3 color images given a dataset
def meanCorrect3Col(data):
    data_copy = data.copy()
    cms = np.nanmean(data_copy,axis=1)[:,np.newaxis]
    data_copy[:,0::3]-= np.nanmean(data_copy[:,0::3],axis=1)[:,np.newaxis]
    data_copy[:,1::3]-= np.nanmean(data_copy[:,1::3],axis=1)[:,np.newaxis]
    data_copy[:,2::3]-= np.nanmean(data_copy[:,2::3],axis=1)[:,np.newaxis]
    return data_copy+cms
    
def candfl_to_fov(fl): return os.path.basename(fl).split('__current')[0]

def get_htags(master_folders):
    folders = ft.flatten([[folder for folder in glob.glob(master_folder+os.sep+'*') 
               if os.path.isdir(folder) and os.path.basename(folder)[0]=='H' and 
               np.any([let in os.path.basename(folder) for let in ['R']]) and
               np.all([let not in os.path.basename(folder)for let in ['H0','dapi']])
              ] for master_folder in master_folders])
    #keep complete sets
    num_daxs = np.array([len(glob.glob(folder+os.sep+'*.dax')) for folder in folders])
    folders = np.array(folders)[num_daxs==np.max(num_daxs)]
    #sort them by the order they were hybed
    folders = np.array(folders)[np.argsort(list(map(io.hybe_number,folders)))]
    htags = list(map(os.path.basename,folders))
    return htags
def get_drift_dic(analysis_folder,fov,htags):
    drift_fl  = analysis_folder+os.sep+fov+r'__drift.pkl'
    txyzs = np.array(pickle.load(open(drift_fl,'rb')))
    txyzs = np.nanmean(txyzs,axis=1)
    txyzs = txyzs-[txyzs[0]]
    drfit_dic = {R:txyz for txyz,R in zip(txyzs,htags)}
    return drfit_dic
def get_all_candidates_in_fov(dic_spots_,R):
    zxys_cands = []
    for ispot_ in dic_spots_:
        dic_spot_,sel_pos_ = dic_spots_[ispot_]
        if R in dic_spot_:
            if len(dic_spot_[R])>0:
                zxys_cands.extend(dic_spot_[R][:,1:4]-drift_)
    zxys_cands = np.array(zxys_cands)
    return zxys_cands
def get_im(data_dax_fl,zxys_cand,icol,cutoff_window = 10):
    min_ = np.min(zxys_cand,axis=0)
    max_ = np.max(zxys_cand,axis=0)
    chr_pt = ((min_+max_)/2).astype(int)
    dax_ = io.dax_im(data_dax_fl,num_col=4,color=icol)
    im,coords = ft.grab_block(dax_,chr_pt,[2*cutoff_window]*len(chr_pt),return_coords=True)
    min_,max_ = zip(*coords)
    return im,min_
def norm_im(im_sm):
    im_medz = np.array(list(map(np.median,im_sm)))[:,np.newaxis,np.newaxis]
    im_sm = im_sm/im_medz
    return im_sm

def return_ims_pts(zxy_,fov_,
                   readouts = None,
                  master_folder = r'\\JACKFRUIT\JackfruitData\DNA_FISH\E20200110_D0021',
                  analysis_folder = r'\\JACKFRUIT\JackfruitData\DNA_FISH\E20200110_D0021\BB_analysis',
                  cols = ['750','647','561'],
                  pix_size = [200,109,109],window=20):

    htags = get_htags([master_folder])
    Rs = np.array([htag+'_'+col_ for htag in htags for col_ in cols])
    if readouts is None: readouts = range(len(zxy_))
    #remaining positions in pixels
    zxy = zxy_/pix_size
    fov = fov_.split('.dax')[0]
    sel_pos = np.nanmedian(zxy,axis=0)
    #dic_spot,sel_pos = dic_spots[ifov][ispot]
    #Load drift
    drfit_dic = get_drift_dic(analysis_folder,fov,htags)
    ims_3d = []
    zxys_im = []
    for iR in tqdm(readouts):
        #extra info about hybe
        R = Rs[iR]
        col = R.split('_')[-1]
        icol = cols.index(col)
        hybe_tag = R.split('_')[0]
        hybe_folder = master_folder+os.sep+hybe_tag
        dax_ = hybe_folder+os.sep+fov+'.dax'
        #drift which is corrected
        drift_ = drfit_dic[hybe_tag]
        #the center position of chromosome 
        sel_pos_ = np.array([sel_pos])+drift_
        #gram image around center position of chr
        #print(dax_)
        im,min_ = get_im(dax_,sel_pos_,icol,cutoff_window = window)
        im = norm_im(im)
        zxys_ = np.array([zxy[iR]-min_+drift_])
        ims_3d.append(im)
        zxys_im.append(zxys_)
    return ims_3d,zxys_im
    
def plot_grid_ims(ims_3d,zxys_im,titles=None,aratio=2,pos_txt=4.5):
    nim = len(ims_3d)
    ncol = int(np.sqrt(nim/aratio))
    nrow= int(np.ceil(float(nim)/ncol))
    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(ncol, nrow),
                     axes_pad=0,
                     )
    if titles is None: titles=list(map(str,np.arange(nim)+1))
    for iax,(ax, im,zxys_,ttl) in enumerate(zip(grid, ims_3d,zxys_im,titles)):
        # Iterating over the grid returns the Axes.
        ax.plot(zxys_[:,2],zxys_[:,1],'o',
                                  markersize=20,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        ax.imshow(np.max(im,axis=0),vmin=0.5,vmax=3.,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0,pos_txt,ttl,color='w')
    plt.show()
    return fig
    
def plot_grid_ims_group(zxys_clean_, fovs_spots_keep, master_folders, scores_clean, analysis_folder, aratio=2,
                                    pos_txt=4.5, saveloc=None, chrs=[], cellid=None, show=False):
    
    # go through all chromosomes with this cell ID.
    for idx, ichr in enumerate(chrs):
        zxy = zxys_clean_[ichr]
        fov = fovs_spots_keep[ichr]

        ims_3d,zxys_im = return_ims_pts(zxy, fov, readouts=None, master_folder=master_folders[0],
                          analysis_folder=analysis_folder, cols=['750','647','561'], pix_size=[200,109,109],window=20)
    
        score_ = np.round(np.log(scores_clean[ichr]),2)
        titles = [str(iR+1)+': '+str(score_[iR]) for iR in np.arange(len(ims_3d))]
        nim = len(ims_3d)
        ncol = int(np.sqrt(nim/aratio))
        nrow= int(np.ceil(float(nim)/ncol))
        fig = plt.figure(figsize=(15, 15))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(ncol, nrow),
                         axes_pad=0,
                         )
        if titles is None: titles=list(map(str,np.arange(nim)+1))
        for iax,(ax, im,zxys_,ttl) in enumerate(zip(grid, ims_3d,zxys_im,titles)):
            # Iterating over the grid returns the Axes.
            ax.plot(zxys_[:,2],zxys_[:,1],'o',
                                      markersize=20,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
            ax.imshow(np.max(im,axis=0),vmin=0.5,vmax=3.,cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0,pos_txt,ttl,color='w')
        
        # only show if requested to.
        if show:
            plt.show()
        
        figname = saveloc+r'grid_ims_cell'+str(cellid)+'_'+str(idx)+'_'+'.png'
        fig.savefig(figname, dpi=300)
        plt.close()
    return fig

'''
This will simply go through all candidates after EM algorithm steps and compute the brightnesses by color for the plot. Assumes
that we have 3 color images.
'''
def brightness_by_color(zxys_f, master_folders, analysis_folder, fov_ids, rnds=42):
    
    #####################################################################################
    ## initializing the files and creating lists of file names for each FOv/hybe, etc. ##
    #####################################################################################
    
    folders = [folder for folder in glob.glob(master_folders[0]+os.sep+'*')
                            if os.path.isdir(folder)
                                and os.path.basename(folder)[0]=='H' # if it starts with H
                                and np.any([pattern in os.path.basename(folder) for pattern in ['R']]) # if it contains any values in ['R']
                                and np.all([pattern not in os.path.basename(folder)for pattern in ['H0','dapi']])] # if it doesnt contain any elements in [ 'H0', 'dapi']
    num_daxs = np.array([len(glob.glob(folder+os.sep+'*.dax')) for folder in folders])
    folders = np.array(folders)[num_daxs==np.max(num_daxs)] # only keep the folders that contain the correct number of dax images.
    #sort them by the order they were hybridized in
    folders = np.array(folders)[np.argsort(map(io.hybe_number,folders))]

    # get the list of fov filenames (shared across all imaging rounds) e.g. Conv_zscan_00.dax
    fovs = np.sort(list(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax'))))

    folders_keep = list(folders[:])

    files = glob.glob(analysis_folder+os.sep+'*__current_cand.pkl')
    fovs = [[os.path.basename(fl).split('__')[0]+'.dax'for fl in files]]
    
    #####################################################################################
    ## loading images, then grabbing block for each spot in each image and computing.  ##
    #####################################################################################
    
    brights = []
    zxy_pix_size = [200,109,109]
    colors = []
    
    zxys_f = np.array(zxys_f)

    for fov_number in tqdm(np.unique(fov_ids)):
        # Get selected chromosomal positions corresponding to the fov_number passed into the script.
        fov_id = fov_number
        fov_filename=fovs[0][fov_id]
        
        # gather dax objects for all imaging rounds (split by color)
        signal_dax_list, signal_color_tags, bead_dax_list, bead_color_tags = io.get_ims_fov_daxmap(folders_keep, fov_filename, col_tags = None, pad = 0)
        # get all spots for a given FOV
        fov_chrs = zxys_f[(np.where(fov_ids == fov_id)[0])]
        
        # go through all images and find all candidates in that image
        for idx,dax_im in enumerate(signal_dax_list[:rnds]):
            
            # go through all chromosomes and grab the block for this dax image in each chromosome
            for chrom in fov_chrs:
                colors.append(idx%3)
                chr_round = np.array(chrom[idx,:])
                center = np.round(np.divide(chr_round,zxy_pix_size),0)
                
                if not np.isnan(center[0]):
                    
                    #grab the data from the center.
                    data = ft.grab_block(dax_im, center, [10]*3)
                    # compute the values
                    brights.append(ft.single_spot_brightness(data))
                    
                else:
                    brights.append(np.nan)
                    
    brights = np.array(brights)
    colors= np.array(colors)
    #removing invalid values from both brightnesses and colors
    brights[brights > 100000] = np.nan
    
    brights_no_nan = np.array(brights)[(np.where(np.logical_not(np.isnan(brights)))[0])]
    colors_no_nan = np.array(colors)[(np.where(np.logical_not(np.isnan(brights)))[0])]
    
    b1 = brights_no_nan[(np.where(colors_no_nan == 0))[0]]
    b2 = brights_no_nan[(np.where(colors_no_nan == 1))[0]]
    b3 = brights_no_nan[(np.where(colors_no_nan == 2))[0]]
    
    plt.figure(figsize=(6,5))
    h1 = plt.hist(b1, bins=1500, alpha=0.6)
    h2 = plt.hist(b2, bins=1500, alpha=0.6)
    h3 = plt.hist(b3, bins=1500, alpha=0.6)
    plt.legend(["750nm", "647nm", "561nm"])
    plt.xlim(0,5000)
    plt.title("Brightness by color for selected spots")
    plt.xlabel("Brightness (AU)")
    plt.ylabel("Counts (# of spots)")
    
    savloc = analysis_folder+os.sep+r"brightness_by_color.pdf"
    plt.savefig(savloc)
    savloc = analysis_folder+os.sep+r"brightness_by_color.png"
    plt.savefig(savloc)
    
    
    return None
    
if __name__ == "__main__":
    from tqdm import tqdm
    #input the folders that store the data
    master_folders = sys.argv[1]
    #define analysis folder
    analysis_folder = sys.argv[2]
    folders, fovs, h0_folder = load_data(master_folders, analysis_folder)
    save_dapi_images(analysis_folder, load_dapi_dax_frame(h0_folder, fovs, 10))
    seg.script_segmentation(analysis_folder)
    save_median_image_across_fovs(folders, analysis_folder)
    flat_field_correction(folders, analysis_folder, fovs)
    get_candidate_spots(analysis_folder, colors=[0,1])
    drift_correction_batch(folders, analysis_folder, jobs=25)
    zxys_f, hs_f, cols_f, scores_f, all_scores_f, dic_spots, dic_drifts, dic_cell_ids, fovs_spots = em_algorithm(analysis_folder)
    make_post_analysis_folder(analysis_folder)
    keep, zxys_clean_filtered, cell_ids_clean, hs_clean, scores_clean, fovs_spots_keep = filter_chromosomes(42, 24, zxys_f, hs_f, cols_f, scores_f, all_scores_f, dic_cell_ids, fovs_spots, -1.5)