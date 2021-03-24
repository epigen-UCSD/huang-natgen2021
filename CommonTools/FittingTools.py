#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

#external packages
import sys,os,glob
import numpy as np

import pickle

import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.optimize import leastsq
from scipy.stats import hmean

from AlignmentTools import *
import IOTools as io
### bleedthrough correction functions
def calc_bleed_matrix(zxys,slopes,order=2):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    slopes a series of multiplicative factors from the regression of small local images
    return m_ a matrix which can be used to calculate the slope for any positionslope
    This m_ is indended to be used with apply_bleed
    """ 
    x_ = np.array(zxys)#
    y_ = np.array(slopes)#y
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    m_ = np.linalg.lstsq(A, y_)[0]
    return m_
def apply_bleed(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions
    m is a matrix computed by function calc_color_mableedtrix
    Returns the slope at that position such that imbleed -= (imreal-1)*slope"""
    if m is None:
        return np.array([0]*len(x))
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    #find the order
    my = len(m)
    assert(my<len(exps))
    #order = int((my-1)/mx)
    
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    slopes = np.dot(A,m)
    return slopes

from scipy.ndimage import zoom
def med_correct(im3d,ksize = 32):
    sz,sxo,syo = im3d.shape
    #assert(sx/ksize==int(sx/ksize) and sy/ksize==int(sy/ksize))
    num_windows_x = int(sxo/ksize)
    num_windows_y = int(syo/ksize)
    sx = num_windows_x*ksize
    sy = num_windows_y*ksize
    im = im3d[:,:sx,:sy]

    im_reshape = im.reshape([sz,num_windows_x,ksize,num_windows_y,ksize])
    im_reshape = np.swapaxes(im_reshape,2,3)
    im_reshape = im_reshape.reshape(list(im_reshape.shape[:-2])+[ksize*ksize])
    im_med = np.median(im_reshape,axis=-1)
    sz,sx_,sy_ = im_med.shape
    
    im_medf = zoom(im_med,[1,float(sxo)/sx_,float(syo)/sy_],order=1)
    return im3d/im_medf
def partition_map(list_,map_):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_)
    map__=np.array(map_)
    return [list(list__[map__==element]) for element in np.unique(map__)]
def gaussian(height,center_z, center_x, center_y, width_z, width_x, width_y, bk=0):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    width_z = float(width_z)
    def gauss(z,x,y):
        g = np.abs(bk)+np.abs(height)*np.exp(
            -(((center_z-z)/width_z)**2+((center_x-x)/width_x)**2+
              ((center_y-y)/width_y)**2)/2.)
        return g
    return gauss
def sphere(center,radius,imshape=None):
    """Returns an int array (size: n x len(center)) with the xyz... coords of a sphere(elipsoid) of radius in imshape"""
    radius_=np.array(radius,dtype=int)
    if len(radius_.shape)==0:
        radius_ = np.array([radius]*len(center),dtype=int)
    xyz = np.array(np.indices(2*radius_+1),dtype=float)
    radius__=np.array(radius_,dtype=float)
    for i in range(len(xyz.shape)-1):
        radius__=np.expand_dims(radius__,axis=-1)
    xyz_keep = np.array(np.where(np.sum((xyz/radius__-1)**2,axis=0)<1))
    xyz_keep = xyz_keep-np.expand_dims(np.array(radius_,dtype=int),axis=-1)+np.expand_dims(np.array(center,dtype=int),axis=-1)
    xyz_keep = xyz_keep.T
    if imshape is not None:
        xyz_keep=xyz_keep[np.all((xyz_keep>=0)&(xyz_keep<np.expand_dims(imshape,axis=0)),axis=-1)]
    return xyz_keep

def grab_block_im(im,center,block_sizes,return_coords=False):
    """Given an n-dim image <im>, a position <center> and a list of sizez <block_sizes>,
    retuns a block of the image of size <block_sizes> from <im> centered at <center>"""
    dims = im.shape
    coords = []
    def in_dim(c,dim):
        c_ = c
        if c_<0: c_=0
        if c_>dim: c_=dim
        return c_
    for c,block,dim in zip(center,block_sizes,dims):
        block_ = int(block//2)
        c=int(c)
        c_min,c_max = in_dim(c-block_,dim),in_dim(c+block-block_,dim)
        coords.append((c_min,c_max))
    slices = [slice(cm,cM) for cm,cM in coords]+[Ellipsis]
    if return_coords:
        return im[slices],coords
    return im[slices]
def grab_block(im,center,block_sizes,return_coords=False):
    """Given an n-dim image <im>, a position <center> and a list of sizez <block_sizes>,
    retuns a block of the image of size <block_sizes> from <im> centered at <center>"""
    if type(im) in [np.core.memmap,np.ndarray]:
        return grab_block_im(im,center,block_sizes,return_coords=return_coords)
        
    dims = im.shape
    coords = []
    def in_dim(c,dim):
        c_ = c
        if c_<0: c_=0
        if c_>dim: c_=dim
        return c_
    for c,block,dim in zip(center,block_sizes,dims):
        block_ = int(block//2)
        c=int(c)
        c_min,c_max = in_dim(c-block_,dim),in_dim(c+block-block_,dim)
        coords.append((c_min,c_max))
    ((zmin,zmax),(ymin,ymax),(xmin,xmax)) = coords
    im_sm = im.get_im(ind_col=None,minx=xmin,maxx=xmax,miny=ymin,maxy=ymax)[zmin:zmax].swapaxes(1,2)
    if return_coords:
        return im_sm,coords
    return im_sm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

'''
This function takes an image from the grab_block function containing only one peak, and computes a local signal 
to noise ratio. Here, I am defining the signal to be the median value of the top 10 pixels and the noise to be the median pixel value in the
image given image. The ratio is of the mean signal minus the mean noise, divided by the noise standard deviation.
'''
def single_spot_local_snr(data, max_spot_area=500):

    data_2d = list(map(np.ravel, data))
    data_1d = np.ravel(data_2d)
    signal_absolute = single_spot_brightness(data) # get the peak height from top x pixels
    
    # compute the noise floor as the median pizel value.
    noise_brightness = np.argsort(data_1d)[:-max_spot_area] # assuming the spot area won't significantly exceed max_spot_area pixels.
    noise_floor = np.nanmedian(data_1d[noise_brightness])
    
    #subtract the noise floor from the signal for a more accurate height of the peak.
    signal_mean = signal_absolute-noise_floor
    
    # compute the stdev in the noise (all data minus the top 80 px as default).
    noise_stdev = np.std(noise_brightness)
    
    # return the ratio of signal mean over deviation in the noise.
    return round(signal_mean/noise_stdev, 3)
    
'''
This function takes an image from the grab_block function containing only one peak and identifies the maximum height of the peak
by ranking the pixel values and taking the median of the top 10 brightest spots.
'''
def single_spot_brightness(data, pixels=30):

    # flatten the data and then take the top 10 brightest pixels. Use their median value as the signal value.
    data_2d = list(map(np.ravel, data))
    data_1d = np.ravel(data_2d)
    
    # take the median of the top 10 pixels. Median will be robust to artifacts muddying the top values.
    top30_brightest = list(np.argsort(data_1d)[-pixels:])
    signal_absolute = np.nanmedian( list([data_1d[i] for i in top30_brightest] ))

    return signal_absolute

'''
This function takes an image from the grab_block function containing only one peak andd fits a gaussian to compute the area of the
peak in the center of this image. This will be done by computing walking outward from the center of the gaussian until we hit
the half max value in 3 directions, then taking the average to compute an average radius.
'''
def single_spot_area(data, pixels=30):
    
    #get the half maximum value
    brightness_abs = single_spot_brightness(data)
    
    # flatten the data and then take the top 10 brightest pixels. Use their median value as the signal value.
    data_2d = list(map(np.ravel, data))
    data_1d = np.ravel(data_2d)
    
    # compute the noise floor as the median pizel value.
    noise_floor = np.nanmedian(data_1d)
    
    #subtract the noise floor from the signal for a more accurate height of the peak.
    signal_mean = brightness_abs-noise_floor
    
    # half max is the half height of the peak above the noise floor of the gaussian.
    half_max = signal_mean/2+noise_floor
    
    area = len(np.where(data_1d > half_max)[0])
    
    return area

def fitmultigaussian(data,centers,radius=10,n_approx=10,width_zxy=[1.,1.,1.],min_width=0.5,fix_width=False):
    """Returns (height, x, y, z, width_x, width_y,width_z,background)
    for the 3D gaussian fit parameters (unconstrained, except for widths>min_width, height>0,background>0) for each point in <centers>
    A spherical neighbourhood of <radius> from the 3d image <data> is used.
    <n_approx> is the list of points in the neighbourhood for which to estimate the paramaters before optimizing for fitting.
    
    Warning: In practice this loosely constrained version is only used to estimate the widths. fitsinglegaussian_fixed_width behaves more robustly
    """
    data_=np.array(data,dtype=float)
    dims = np.array(data_.shape)
    xyz_unq=set()
    params=[]
    for center in centers:
        xyz = sphere(center,radius,imshape=dims)
        for xyz_ in xyz:
            xyz_unq.add(tuple(xyz_))
        data__=data_[xyz.T[0],xyz.T[1],xyz.T[2]]
        bk = np.median(np.sort(np.ravel(data__))[:n_approx])
        height = (np.median(np.sort(np.ravel(data__))[-n_approx:])-bk)
        center_z,center_x,center_y = center
        width_z,width_x,width_y = np.array(width_zxy)-min_width
        if fix_width:
            params_ = (height,center_z,center_x,center_y,bk)
        else:
            params_ = (height,center_z,center_x,center_y,width_z,width_x,width_y,bk)
        params.append(params_)
    params = np.array(params)
    xyz_unq = np.array([val for val in xyz_unq]).T
    dist_bk = cdist(xyz_unq.T,centers)
    dist_bk[dist_bk<1]=1
    weigh_bk = dist_bk/np.expand_dims(np.sum(dist_bk,axis=-1),-1)
    data_=data_[xyz_unq[0],xyz_unq[1],xyz_unq[2]]
    def gaussian(height,center_z, center_x, center_y, width_z=width_zxy[0]-min_width, 
                 width_x=width_zxy[1]-min_width, 
                 width_y=width_zxy[2]-min_width, 
                 bk=0,min_width=min_width):
        """Returns a gaussian function with the given parameters"""
        
        width_x_ = np.abs(width_x)+float(min_width)
        width_y_ = np.abs(width_y)+float(min_width)
        width_z_ = np.abs(width_z)+float(min_width)
        height_ = np.abs(height)
        bk_ = np.abs(bk)
        def gauss(z,x,y):
            g = bk_+height_*np.exp(
                -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                  ((center_y-y)/width_y_)**2)/2.)
            return g
        return gauss
    def errorfunction(p):
        p_ = np.reshape(p,[len(centers),-1])
        bk_map = np.dot(weigh_bk,np.abs(p_[:,-1]))
        f=bk_map+np.sum([gaussian(*p__)(*xyz_unq) for p__ in p_[:,:-1]],0)
        g=data_
        #err=np.ravel(f-g-g*np.log(f/g))
        err=np.ravel(f-g)
        #print np.mean(err**2)
        return err
    p, success = leastsq(errorfunction, params)
    p = np.reshape(p,[len(centers),-1])
    p=np.abs(p)
    #p[:1:4]+=0.5
    if fix_width:
        p = np.concatenate([p[:,:-1],[width_zxy]*len(p),np.expand_dims(p[:,-1],-1)],axis=-1)
    else:
        p[:,4:7]+=min_width
    return  np.reshape(p,[len(centers),-1]),success

def gauss_ker(sig_xyz=[2,2,2],sxyz=16,xyz_disp=[0,0,0]):
    dim = len(xyz_disp)
    xyz=np.indices([sxyz+1]*dim)
    for i in range(len(xyz.shape)-1):
        sig_xyz=np.expand_dims(sig_xyz,axis=-1)
        xyz_disp=np.expand_dims(xyz_disp,axis=-1)
    im_ker = np.exp(-np.sum(((xyz-xyz_disp-sxyz/2.)/sig_xyz**2)**2,axis=0)/2.)
    return im_ker
def add_source(im_,pos=[0,0,0],h=200,sig=[2,2,2]):
    pos = np.array(pos)+0.5
    im=np.array(im_,dtype=float)
    pos_int = np.array(pos,dtype=int)
    xyz_disp = -pos_int+pos
    im_ker = gauss_ker(sig_xyz=sig,sxyz=int(np.max(sig)*5),xyz_disp=xyz_disp)
    im_ker_sz = np.array(im_ker.shape,dtype=int)
    pos_min = pos_int-im_ker_sz//2
    pos_max = pos_min+im_ker_sz
    im_shape = np.array(im.shape)
    def in_im(pos__):
        pos_=np.array(pos__,dtype=int)
        pos_[pos_>=im_shape]=im_shape[pos_>=im_shape]#-1
        pos_[pos_<0]=0
        return pos_
    pos_min_ = in_im(pos_min)
    pos_max_ = in_im(pos_max)
    pos_min_ker = pos_min_-pos_min
    pos_max_ker = im_ker_sz+pos_max_-pos_max
    #print zip(pos_min_ker,pos_max_ker),zip(pos_min_,pos_max_),zip(pos_min,pos_max)
    slices_ker = [slice(pm,pM)for pm,pM in zip(pos_min_ker,pos_max_ker)]
    slices_im = [slice(pm,pM)for pm,pM in zip(pos_min_,pos_max_)]
    im[slices_im]+=im_ker[slices_ker]*h
    return im
def subtract_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=-pfit[0],sig=pfit[-3:])
def plus_source(im,pfit):
    return add_source(im,pos=pfit[1:4],h=pfit[0],sig=pfit[-3:])
def fitsinglegaussian_fixed_width(data,center,radius=5,n_approx=10,width_zxy=[1.,1.,1.]):
    """Returns (height, x, y,z, width_x, width_y,width_z,bk)
    for the 3D gaussian fit for <radius> around a 3Dpoint <center> in the 3Dimage <data>
    <width_zxy> are the widths of the gaussian
    """
    data_=data
    dims = np.array(data_.shape)
    if center is  not None:
        center_z,center_x,center_y = center
    else:
        xyz = np.array(list(map(np.ravel,np.indices(data_.shape))))
        data__=data_[xyz[0],xyz[1],xyz[2]]
        args_high = np.argsort(data__)[-n_approx:]
        center_z,center_x,center_y = np.median(xyz[:,args_high],axis=-1)
    
    xyz = sphere([center_z,center_x,center_y],radius,imshape=dims).T
    if len(xyz[0])>0:
        data__=data_[xyz[0],xyz[1],xyz[2]]
        sorted_data = np.sort(data__)#np.sort(np.ravel(data__))
        bk = np.median(sorted_data[:n_approx])
        height = (np.median(sorted_data[-n_approx:])-bk)
            
        width_z,width_x,width_y = np.array(width_zxy)
        params_ = (height,center_z,center_x,center_y,bk)
        
        def gaussian(height,center_z, center_x, center_y,
                     bk=0,
                     width_z=width_zxy[0], 
                     width_x=width_zxy[1], 
                     width_y=width_zxy[2]):
            """Returns a gaussian function with the given parameters"""
            width_x_ = np.abs(width_x)
            width_y_ = np.abs(width_y)
            width_z_ = np.abs(width_z)
            height_ = np.abs(height)
            bk_ = np.abs(bk)
            def gauss(z,x,y):
                g = bk_+height_*np.exp(
                    -(((center_z-z)/width_z_)**2+((center_x-x)/width_x_)**2+
                      ((center_y-y)/width_y_)**2)/2.)
                return g
            return gauss
        def errorfunction(p):
            f=gaussian(*p)(*xyz)
            g=data__
            #err=np.ravel(f-g-g*np.log(f/g))
            err=np.ravel(f-g)
            return err
        p, success = leastsq(errorfunction, params_)
        p=np.abs(p)
        p = np.concatenate([p,width_zxy])
        return  p,success
    else:
        return None,None
def fast_local_fit(im,centers,radius=5,width_zxy=[1,1,1],return_good=False):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds for <radius> and with fixed <width_zxy>
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    """
    ps=[]
    good = []
    im_=np.array(im)
    for center in centers:
        p,success = fitsinglegaussian_fixed_width(im_,center,radius=radius,n_approx=5,width_zxy=width_zxy)
        good.append(False)
        if p is not None:
            if np.max(np.abs(p[1:4]-center))<radius:
                ps.append(p)
                good[-1]=True
    if return_good:
        return np.array(ps),np.array(good,dtype=bool)
    return np.array(ps)
def fit_seed_points_base(im,centers,width_zxy=[1.,1.,1.],radius_fit=5,n_max_iter = 10,max_dist_th=0.25):
    """
    Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds (in order of brightness) and subtract the gaussian signal.
    Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
    Warning: Generally a bit slow. In practice, the faster version fast_local_fit is used.
    """
    #print "Fitting:" +str(len(centers[0]))+" points"
    z,x,y = centers
    
    
    if len(x)>0:
        #get height of the points and order by brightness
        h = [im[int(z_),int(x_),int(y_)] for z_,x_,y_ in zip(z,x,y)]
        inds = np.argsort(h)[::-1]
        z,x,y = z[inds],x[inds],y[inds]
        zxy = np.array([z,x,y],dtype=int).T
        
        #fit the points in order of brightness and at each fit subtract the gaussian signal
        ps = []
        im_subtr = np.array(im,dtype=float)
        for center in zxy:
            p,success = fitsinglegaussian_fixed_width(im_subtr,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
            if p is not None:
                ps.append(p)
                im_subtr = subtract_source(im_subtr,p)

        im_add = np.array(im_subtr)

        max_dist=np.inf
        n_iter = 0
        while max_dist>max_dist_th:
            ps_1=np.array(ps)
            ps_1=ps_1[np.argsort(ps_1[:,0])[::-1]]
            ps = []
            ps_1_rem=[]
            for p_1 in ps_1:
                center = p_1[1:4]
                im_add = plus_source(im_add,p_1)
                p,success = fitsinglegaussian_fixed_width(im_add,center,radius=radius_fit,n_approx=5,width_zxy=width_zxy)
                if p is not None:
                    ps.append(p)
                    ps_1_rem.append(p_1)
                    im_add = subtract_source(im_add,p)
            ps_2=np.array(ps)
            ps_1_rem=np.array(ps_1_rem)
            dif = ps_1_rem[:,1:4]-ps_2[:,1:4]
            max_dist = np.max(np.sum(dif**2,axis=-1))
            n_iter+=1
            if n_iter>n_max_iter:
                break
        return ps_2
    else:
        return np.array([])
def get_ims_meds(folder,num_cols=None,fovs=None,median_fl=None,overwrite=False,tag='mid',verbose=False,func=np.median):
    """Given a folder and a list of FOVs in that folder (i.e. ['Conv_zscan_00.dax', 'Conv_zscan_01.dax',...]) 
    this computes the median image (2d) across all the FOVs.
    tag can be 'mid','start','all'
    """
    
    compute=True
    if median_fl is not None and not overwrite:
        if os.path.exists(median_fl):
            im_meds=np.load(median_fl)
            compute=False
    if compute:
        if num_cols is None:
            num_cols = folder.count(',')+2
        imst_cols=[]
        if fovs is None:
            fovs = list(map(os.path.basename,glob.glob(folder+os.sep+'*.dax')))
        fovs_ = fovs
        if verbose:
            from tqdm import tqdm_notebook as tqdm
            fovs_ = tqdm(fovs)
        for fov in fovs_:
            ims_ = io.dax_im(folder+os.sep+fov,num_col=num_cols).get_mids(tag=tag)
            imst_cols.append(ims_)
        im_meds=np.array(func(imst_cols,axis=0),dtype=np.uint16)
        if median_fl is not None:
            np.save(median_fl,im_meds)
    return im_meds
from scipy.ndimage.filters import maximum_filter,minimum_filter,median_filter,gaussian_filter

def get_seed_points_base(im,gfilt_size_min=1,gfilt_size_max=3,filt_size=3,th_seed=0.,th_std=0.,max_num=None,
                         use_snr=False,hot_pix_th=0,return_h=False):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius 3 above th_seed from the minimum
    #3 remove hot pixels (high multiplicity of xy positions with different z)
    """
    im_plt=np.array(im,dtype=np.float32)
    #if gfilt_size>0:
    #    im_plt = gaussian_filter(im,gfilt_size)
    max_filt = maximum_filter(im_plt,filt_size)
    min_filt = minimum_filter(im_plt,filt_size)
    g_filt = gaussian_filter(im_plt,gfilt_size_max)
    g_filt_sm = gaussian_filter(im_plt,gfilt_size_min)
    im_plt2 = (max_filt==im_plt)&(min_filt!=im_plt)
    z,x,y = np.where(im_plt2)
    h = g_filt_sm[z,x,y]-g_filt[z,x,y]
    if th_std>0:
        h_all = g_filt_sm-g_filt
        h_mn,h_std = np.mean(h_all),np.std(h_all)
        keep = (h-h_mn)>h_std*th_std
        x,y,z,h = x[keep],y[keep],z[keep],h[keep]
    snr = 1.*g_filt_sm[z,x,y]/g_filt[z,x,y]
    
    if use_snr:
        keep = snr>th_seed
    else:
        keep = h>th_seed
    x,y,z = x[keep],y[keep],z[keep]
    h,snr = h[keep],snr[keep]
    #get rid of hot pixels
    if hot_pix_th>0 and len(x)>0:
        xy = y*np.max(x)+x
        xy_,cts_ = np.unique(xy,return_counts=True)
        bad_xy = xy_[cts_>hot_pix_th]
        keep = np.array([xy_ not in bad_xy for xy_ in xy],dtype=bool)
        x,y,z = x[keep],y[keep],z[keep]
        snr=snr[keep]
        h = h[keep]
    centers = np.array([z,x,y])
    #sort by absolute brightness or signal to noise ratio (snr)
    if not use_snr:
        ind = np.argsort(h)[::-1]
    else:
        ind = np.argsort(snr)[::-1]
    centers = np.array([z[ind],x[ind],y[ind]])
    if return_h:
        centers = np.array([z[ind],x[ind],y[ind],h[ind]])
    if max_num is not None:
        centers = centers[:,:max_num]
    return centers
def update_candid_spot_old(im,low_limit,prefered_limit,chr_pts_,txyz_,tag,cutoff_window=20,cutoff_chr=10,candid_spot={},fl_cands=None):
    """im -> 3dimage of the current hybe
    chr_pts_ -> the postions of the selected chromosomes
    txyz_ -> drift from current hybe to reference frame where chromosome was selected 
             and to which the drift was computed to
    tag -> the tag on how to update candid_spot dictionary
    candid_spot -> the dictionary where to save data
    fl_cands -> the file where to save data
    """
    [z,x,y,h]=get_seed_points_base(im,gfilt_size=0.75,filt_size=3,th_seed=low_limit,hot_pix_th=4,return_h=True)
    centersh = np.array([z,x,y,h]).T
    centers = np.array([z,x,y]).T
    #compare with prefered spots
    dists = cdist(chr_pts_,centers-[txyz_])#distances to chromosomes should be computed in the reference where the chromosomes where selected (ref=0)
    mindist = np.min(dists,0)
    keep = mindist<cutoff_window #keep only points within a distance cutoff_window from a chromosome
    centersh_close = centersh[keep] 
    closest_chr = np.argmin(dists[:,keep],0) #vector with ids of closest chromosomes
    centersh_close = np.concatenate([centersh_close.T,[mindist[keep]]],axis=0).T #extend with distance from closest chromosome
    chroms_id = np.unique(closest_chr)
    partitioned_info = partition_map(centersh_close,closest_chr) #partition to chromosome
    
    #iterate through the crhomosomes in the field of view
    for pts_info,chr_id in zip(partitioned_info,chroms_id):
        z_,x_,y_,h_,dist_=np.array(pts_info).T
        centes_fit=np.array([z_,x_,y_]).T
        keep_close=dist_<cutoff_chr
        if np.sum(keep_close)>0:
            if np.max(h_[keep_close])>=prefered_limit:
                centes_fit=centes_fit[h_>=prefered_limit]
        pp = fast_local_fit(im,centes_fit)
        if len(pp)>0:
            pp[:,1:4]=pp[:,1:4]-[txyz_]#apply drift correction
        #record in dictionary in the old format
        candid_dic,_ = candid_spot.get(chr_id,({},None))
        candid_dic[tag]=pp#drif corrected
        candid_spot[chr_id]=(candid_dic,chr_pts_[chr_id])
    if fl_cands is not None:
        pickle.dump(candid_spot,open(fl_cands,'wb'))
        
def linrescale(im_mean1,im_mean2,n=100000):
    """Given 2 images this tranforms image 2 to be linearly rescaled to image 1 pixelwize"""
    indsr = np.random.randint(0,len(im_mean1.ravel()),n)
    x,y = im_mean1.ravel()[indsr],im_mean2.ravel()[indsr]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return (im_mean2-c)/m
import cv2
def normalzie_im(im,sz=20):
    im_ = np.array(im,dtype=np.float32)
    im_blur = np.array([cv2.blur(im__,(sz,sz)) for im__ in im_],dtype=np.float32)
    im_ =im_/im_blur-1
    return im_
def blurnorm2d(im,gb):
    """Normalize an input 2d image <im> by dividing by a cv2 gaussian filter of the image"""
    im_ = im.astype(np.float32)
    blurred = cv2.blur(im_,(gb,gb))
    return im_/blurred
def get_seed_points_base_v2(im_sm,gfilt_size=5,filt_size=3,th_seed=3.,max_num=None):
    """Get the seed points in an image.
    #1 perform a gaussian filter
    #2 find local maxima within a radius <filt_size> with <th_seed> above local maximum (in log space)
    """
    if gfilt_size!=0: im_norm = normalzie_im(im_sm,gfilt_size)
    else: im_norm=im_sm
    std_ = np.nanstd(im_norm)
    hcutoff = std_*th_seed
    #print std_,np.max(im_norm)
    z,x,y = np.where(im_norm>hcutoff)
    h_im = im_norm[z,x,y]
    sz,sx,sy = im_norm.shape
    keep = h_im>0
    pix = int(filt_size/2)
    deltas = range(-pix,pix+1)
    for deltax in deltas:
        for deltay in deltas:
            for deltaz in deltas:
                #if (np.abs(deltax)+np.abs(deltay)+np.abs(deltaz))>0:
                keep &= (h_im>=im_norm[(z+deltaz)%sz,(x+deltax)%sx,(y+deltay)%sy])
    zf,xf,yf = z[keep],x[keep],y[keep]
    hf = im_norm[zf,xf,yf]
    inds = np.argsort(hf)[::-1]
    centers_zxyh = np.array([zf[inds],xf[inds],yf[inds],hf[inds]])
    if max_num is not None:
        centers_zxyh = centers_zxyh[:,:max_num]
    return centers_zxyh,std_

def get_2brighspots_per_cell(im_segm,fovnum,zxyh0,dist = 8):
    """Given a 2d image <im_segm> with cells of different index,and a set of z,x,y,h spot coordinates and brightnss <zxyh0>
    This returns the 2 brightest spots that are at least 10 pixels away"""
    zxyh0_keep = []
    z,x,y,h = zxyh0
    xy = (x+10000*y).astype(int) # convert to an int to speed up comparisons
    zxyh0_ = np.array(zxyh0).T
    cell_ids = []
    # each cell is numbered 1 to num_cells (all pixels in a given cell are equal to its number)
    for icell in range(1,np.max(im_segm)+1):
        cell_xy = np.where(im_segm==icell) # returns indices where the mask is the cell id.
        cell_xy = cell_xy[0]+10000*cell_xy[1] # convert to an int to speed up comparisons
        
        keep = np.in1d(xy,cell_xy) # this retrieves candidates within the cell (xy positions as one val)

        #these are all the candidate points within the cell
        zxyh0_c = zxyh0_[keep]
        
        #get the brightest point
        if len(zxyh0_c)>0:
            i1 = np.argmax(zxyh0_c[:,-1])
            zxy1 = zxyh0_c[i1,:3]
            zxyh0_keep.append(zxyh0_c[i1])
            #get the second brightest point that is at least <dist> pixels away 
            zxyh0_cl = zxyh0_c[np.linalg.norm(zxyh0_c[:,:3]-zxy1,axis=-1)>dist]
            if len(zxyh0_cl)>0:
                i2 = np.argmax(zxyh0_cl[:,-1])
                zxyh0_keep.append(zxyh0_cl[i2])
            else:
                print("one spot found only!")
                zxyh0_keep = zxyh0_keep[:-1]
                continue #zxyh0_keep.append([np.nan]*4)
        else:
            print("no spots found!")
            continue #zxyh0_keep.extend([[np.nan]*4]*2)
            
        # save cell ids for the spots we identify. (2 per cell here)
        # to differentiate between FOVS, we number them by 1000*fov*cellnum since we wont have
        # more than 1000 cells per fov ever.
        cell_ids.append((1000*fovnum)+icell)
        cell_ids.append((1000*fovnum)+icell)
        
    zxyh0_keep = np.array(zxyh0_keep).T
    
    return zxyh0_keep, cell_ids

def return_candidate_centers(im,dist_th=20,ex=20,ey=20,ez=2,gmax=5,filtsz=3,ths=4,plt_val=False,save_file=None):
    zs,xs,ys = im.shape
    centersh,std_ = get_seed_points_base_v2(im,gfilt_size=gmax,filt_size=filtsz,th_seed=ths,max_num=None)
    z,x,y,h = centersh#sparse_centers(centersh, dist_th=dist_th, brightness_th=0, max_num=np.inf)
    keep = (z<zs-ez)&(z>ez)&(x<xs-ex)&(x>ex)&(y<ys-ey)&(y>ey)
    z,x,y,h = z[keep],x[keep],y[keep],h[keep]
    dic = {'zxyh_full':np.array(centersh).T,'zxyh':np.array([z,x,y,h]).T}
    if save_file is not None:
        pickle.dump(dic,open(save_file,'wb'))
    if plt_val:
        plt.figure()
        plt.plot(y,x,'rx')
        plt.imshow(np.max(im[:],0),cmap='gray',vmax = np.median(im[z.astype(int),x.astype(int),y.astype(int)]))
        plt.show()    
    return z,x,y,h
def get_intersections(zxyhs,absd = 15):
    ref = zxyhs[0][:,:3]
    dsts,inds = [],[]
    for i in range(0,len(zxyhs)):
        target = zxyhs[i][:,:3]
        M = cdist(ref,target)
        ind = np.argmin(M,axis = -1)
        dst = np.min(M,axis = -1)
        dsts.append(dst)
        inds.append(ind)
    keep = np.all(np.array(dsts)<absd,axis=0)
    inds = np.array(inds)[:,keep]
    zxys = [zxyh_[ind_,:3] for zxyh_,ind_ in zip(zxyhs,inds)]
    zxys_final = np.mean(np.array(zxys),0)
    return zxys_final    
def update_candid_spot(im,chr_pts_,txyz_,tag,th_std=2,num_pts=5,
                        cutoff_window=20,cutoff_chr=10,dic_bleed={},candid_spot={},drift_dic={},fl_cands=None,im_repair=None,plt_val=False,verbose=True):
    """im -> 3dimage of the current hybe
    chr_pts_ -> the postions of the selected chromosomes
    txyz_ -> drift from current hybe to reference frame where chromosome was selected 
             and to which the drift was computed to
    tag -> the tag on how to update candid_spot dictionary
    candid_spot -> the dictionary where to save data
    fl_cands -> the file where to save data
    """
    drift_dic[tag]=txyz_
    
    for ichr,chr_pt in enumerate(chr_pts_):
        im_sm,coords = grab_block(im,chr_pt+txyz_,[2*cutoff_window]*len(chr_pt),return_coords=True)
        pp = []
        
        # make sure we grabbed a block with all dims > 0
        if np.prod(im_sm.shape)>0:
        
            # do flat field corr
            im_sm = np.array(im_sm,dtype=np.float32)
            slices = [slice(cm,cM) for cm,cM in coords]+[Ellipsis]
            if im_repair is not None:
                im_sm = im_sm/im_repair[slices[1:]].astype(float)
            im_medz = np.array(list(map(np.median,im_sm)))[:,np.newaxis,np.newaxis]
            im_sm = im_sm/im_medz
            zxy_cm=np.array([[(cm+cM)/2. for cm,cM in coords]])
            
            #correct for bleedthrough
            if len(dic_bleed)>0:
                #get bleedthrough images and normalize them
                im_bs,mbs = dic_bleed['ims'],dic_bleed['m_slopes']
                im_bs = [grab_block(im_b,chr_pt+txyz_,[2*cutoff_window]*len(chr_pt),return_coords=False) for im_b in dic_bleed['ims']]
                im_bs_ = []
                for im_b in im_bs:
                    if im_repair is not None:
                        im_b = im_b/im_repair[slices[1:]].astype(float)
                    im_medz = np.array(list(map(np.median,im_b)))[:,np.newaxis,np.newaxis]
                    im_b = im_b/im_medz
                    im_bs_.append(im_b)
                #ensure same dimension
                min_sz = np.min([len(im_b) for im_b in list(im_bs_)+[im_sm]])
                im_bs_ = [im_b[:min_sz] for im_b in list(im_bs_)]
                im_sm = im_sm[:min_sz]
                #calculate total bleed and subtract
                im_total_bleed = np.sum([(im_b-1)*apply_bleed(zxy_cm,m=mb)[0] for mb,im_b in zip(mbs,im_bs_)],0)
                if plt_val:
                    plt.figure()
                    plt.title(tag+'_chr:'+str(ichr)+'-prebleed')
                    im_plt = np.max(im_sm,0)
                    plt.imshow(im_plt)
                    plt.colorbar()
                    plt.show()
                    
                    plt.figure()
                    plt.title(tag+'_chr:'+str(ichr)+'-bleed')
                    im_plt = np.max(im_total_bleed,0)
                    plt.imshow(im_plt)
                    plt.colorbar()
                    plt.show()
                im_sm -= im_total_bleed
                print("Total bleed: ", im_total_bleed)
            mins = np.array(coords)[:,0]
            [z,x,y,h]=get_seed_points_base(im_sm,gfilt_size_min=1,gfilt_size_max=3,filt_size=3,th_seed=0,th_std=th_std,max_num=None,
                                 use_snr=False,hot_pix_th=0,return_h=True)
            centersh = np.array([z,x,y,h]).T
            centers = np.array([z,x,y]).T
            counter = 0
            centers_keep = []
            counter_close = 0
            while True:
                if counter>len(centers)-1:
                    break
                center = centers[counter]
                center_ = center+mins-txyz_
                dists = cdist(chr_pts_,[center_])[:,0]
                dist =  np.linalg.norm(center_-chr_pt)
                if ichr==np.argmin(dists) and dist<cutoff_window:
                    #accept
                    centers_keep.append(center)
                    if dist<cutoff_chr:
                        counter_close+=1
                counter+=1
                if counter_close>=num_pts:
                    break

            pp = fast_local_fit(im_sm,centers_keep)
            
            ###plot for testing
            if plt_val:
                plt.figure()
                plt.title(tag+'_chr:'+str(ichr))
                im_plt = np.max(im_sm,0)
                plt.imshow(im_plt)
                plt.colorbar()
                chr_pts__ = chr_pts_-mins
                plt.plot(chr_pts__[:,2],chr_pts__[:,1],'bo')
                plt.plot(chr_pts__[[ichr],2],chr_pts__[[ichr],1],'ro')
                if len(pp)>0:
                    plt.plot(pp[:,3],pp[:,2],'rx')
                plt.xlim(0,im_plt.shape[0])
                plt.ylim(0,im_plt.shape[1])
                plt.show()
            
            if len(pp)>0:
                pp[:,1:4]=pp[:,1:4]+[mins-txyz_]#apply drift correction
            #record in dictionary in the old format
        candid_dic,_ = candid_spot.get(ichr,({},None))
        candid_dic[tag]=pp#drif corrected
        candid_spot[ichr]=(candid_dic,chr_pts_[ichr])
    if fl_cands is not None:
        pickle.dump((candid_spot,drift_dic),open(fl_cands,'wb'))
def sparse_centers(centersh,dist_th=0,brightness_th=0,max_num=np.inf):
    """assuming input = zxyh"""
    all_cents = np.array(centersh).T
    centers = [all_cents[0]]
    from scipy.spatial.distance import cdist
    counter=0
    while True:
        counter+=1
        if counter>len(all_cents)-1:
            break
        if all_cents[counter][-1]<brightness_th:
            break
        dists = cdist([all_cents[counter][:3]],[c[:3] for c in centers])
        if np.all(dists>dist_th):
            centers.append(all_cents[counter])
        if len(centers)>=max_num:
            break
    return np.array(centers).T
## Bead drift correction functions
        
def get_STD_centers(im,hseed=0.,nseed=100,plt_val=False):
    """Given an image <im> a min height <hseed> and a number of seeds <nseed> this returns  zxy positions of the  brightest ~<nseed> fits.
    This is intended for beads.
    """
    im_=np.array(im,dtype=np.float32)
    
    centers = get_seed_points_base(im_,gfilt_size_min=1,gfilt_size_max=3,filt_size=3,th_seed=hseed,max_num=nseed,
                         use_snr=False,hot_pix_th=0,return_h=False)
    centers = np.array(centers).T
    pfits = fast_local_fit(im,centers,radius=5,width_zxy=[1,1,1])
    if len(pfits)==0:
        cents = np.array([[np.inf]*3])
    else:
        cents = pfits[:,1:4]
    #cents = centers.T
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im,0),interpolation='nearest')
        plt.plot(cents[:,2],cents[:,1],'o')
        plt.show()
    return cents
    
def simplify_txyz(txyz_both):
    """
    ensure that the final output is Nx(tx,ty,tz) for the list of drifts
    """
    if len(np.array(txyz_both).shape)==3:
        txyz = np.mean(txyz_both,1)
    else:
        txyz = txyz_both
    return txyz
from scipy.spatial.distance import cdist
def get_ref_pts(im_ref_sm,dist_th = 5,nbeads = 400):
    #fit reference
    z,x,y,h = get_seed_points_base(im_ref_sm,return_h=True,th_std=4)
    zk,xk,yk,hk = sparse_centers((z,x,y,h),dist_th=dist_th,brightness_th=0,max_num=nbeads)
    cr1 = np.array([zk,xk,yk]).T
    pfits1 = fast_local_fit(im_ref_sm,cr1,radius=5,width_zxy=[1,1,1])
    cr1 = pfits1[:,1:4]
    return cr1
def get_cand_pts(im_sm,cr1,tzxy,dist_th = 5):
    #fit candidate
    z,x,y,h = get_seed_points_base(im_sm,return_h=True,th_std=4)
    cr2_ = np.array([z,x,y]).T
    cr2_cand = cr2_+tzxy

    M = cdist(cr1,cr2_cand)
    M_th = M<=dist_th
    pairs = [(_cr1,cr2_[m][np.argmax(h[m])]) for _cr1,m in zip(cr1,M_th) if np.sum(m)>0]
    cr1,cr2 = list(map(np.array,zip(*pairs)))
    pfits2,keep = fast_local_fit(im_sm,cr2,radius=5,width_zxy=[1,1,1],return_good=True)
    cr2 = pfits2[:,1:4]
    return np.array(cr1)[keep],cr2
def get_STD_beaddrift_v2(ims_beads,coord_sel=None,sz_ex=50,desired_nbeads = 20, 
                      desired_displ = 0.4, hseed=150,nseed=100,
                      ref=None,force=False,save=True,save_file='temp.pkl',verbose=True):
    """Given a list of bead images <ims_beads> this handles the fine bead drift correction.
    For each 3d image in <ims_beads> the beads for subimages of size <sz_ex>,
    centered at [center,center,center],[center,center,center]+[0,2*sz_ex,0] are fitted using #get_STD_centers with paramaters <hseed>,<nseed>.
    Beads for each of the two subimages are aligned with the corresponding beads for the reference image of index <ref> (default = len(ims_beads)/2) in ims_beads.
    """
    repeat = True
    txyzs_both = []
    txyzs_both_med = []
    bad_inds = []
    if save:
        save_cor = save_file
        if os.path.exists(save_cor):
            txyzs_both = pickle.load(open(save_cor,'rb'))
            if len(txyzs_both)==len(ims_beads) and not force:
                repeat=False
                return txyzs_both,repeat
    if force:
        txyzs_both = []
        bad_inds = []
    if repeat:
        #get txyz
        if ref is None: ref=len(ims_beads)//2
        im_ref = ims_beads[ref]
        coord_sel = np.array(im_ref.shape)//2
        coord_sel1 =coord_sel
        im_ref_sm1,coords1 = grab_block(im_ref,coord_sel1,[sz_ex]*3,return_coords=True)
        cents_ref1 = get_ref_pts(im_ref_sm1,dist_th = 5,nbeads = nseed)#get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,-sz_ex,0])+coord_sel
        im_ref_sm2,coords2 = grab_block(im_ref,coord_sel2,[sz_ex]*3,return_coords=True)
        cents_ref2 = get_ref_pts(im_ref_sm2,dist_th = 5,nbeads = nseed)#get_STD_centers(im_ref_sm2,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 2
        
        cutoff_=2
        xyz_res_=1
        for iim in bad_inds+list(range(len(txyzs_both),len(ims_beads))):
            
            if verbose:print("Aligning "+str(iim+1))
            im = ims_beads[iim]
            txy_prev = np.array([0,0,0])
            th_good_bead = 2#pixels
            #set1
            im_sm1,coords1 = grab_block(im,coord_sel1+txy_prev,[sz_ex]*3,return_coords=True)
            Tzxy = fft3d_from2d(im_ref_sm1,im_sm1,gb=5,max_disp=np.inf)
            cr1,cr2 = get_cand_pts(im_sm1,cents_ref1,Tzxy,dist_th = 5)
            txyz1 = np.median(cr2-cr1,axis=0)
            c11 = np.sum(np.linalg.norm(cr2-cr1-txyz1,axis=-1)<th_good_bead)#sub pixel number of beads
            #set2
            im_sm2,coords2 = grab_block(im,coord_sel2+txy_prev,[sz_ex]*3,return_coords=True)
            Tzxy = fft3d_from2d(im_ref_sm2,im_sm2,gb=5,max_disp=np.inf)
            cr1,cr2 = get_cand_pts(im_sm2,cents_ref2,Tzxy,dist_th = 5)
            txyz2 = np.median(cr2-cr1,axis=0)
            c21 = np.sum(np.linalg.norm(cr2-cr1-txyz2,axis=-1)<th_good_bead)#sub pixel number of beads
            
            txyz = (txyz1+txyz2)/2.
            if verbose: print(txyz1,txyz2,c11,c21)
            displ = np.max(np.abs(txyz1-txyz2))
            if (displ>desired_displ) or (c11<desired_nbeads) or (c21<desired_nbeads):
                if verbose: print("Suspecting failure.")
                #set3
                coord_sel3 = np.array([0,sz_ex,0])+coord_sel
                im_ref_sm3,coords3 = grab_block(im_ref,coord_sel3,[sz_ex]*3,return_coords=True)
                cents_ref3 = get_ref_pts(im_ref_sm3,dist_th = 5,nbeads = nseed)
                im_sm3,coords3 = grab_block(im,coord_sel3+txy_prev,[sz_ex]*3,return_coords=True)
                Tzxy = fft3d_from2d(im_ref_sm3,im_sm3,gb=5,max_disp=np.inf)
                cr1,cr2 = get_cand_pts(im_sm3,cents_ref3,Tzxy,dist_th = 5)
                txyz3 = np.median(cr2-cr1,axis=0)
                c31 = np.sum(np.linalg.norm(cr2-cr1-txyz3,axis=-1)<th_good_bead)#sub pixel number of beads
                
                measures = list(map(np.sum,list(map(np.abs,[txyz3-txyz1,txyz3-txyz2,txyz1-txyz2]))))
                imeasure = np.argmin(measures)
                nbds = [c11,c21,c31]
                nbds_variants = [[nbds[2],nbds[0]],[nbds[2],nbds[1]],[nbds[0],nbds[1]]]
                variants = [[txyz3,txyz1],[txyz3,txyz2],[txyz1,txyz2]]
                best_pair = variants[imeasure]
                best_measure = measures[imeasure]
                
                if best_measure>6*desired_displ or np.max(nbds_variants[imeasure])<desired_nbeads:
                    best_pair = [[txyz1,txyz1],[txyz2,txyz2],[txyz3,txyz3]][np.argmax(nbds)]
                
                if verbose: print(best_pair,measures[imeasure],nbds)
                
                
            else:
                best_pair = [txyz1,txyz2]
            #update txyzs_both
            if iim<len(txyzs_both):
                txyzs_both[iim]= best_pair
            else:
                txyzs_both.append(best_pair)
        if save:
            save_cor = save_file
            pickle.dump(txyzs_both,open(save_cor,'wb'))
    return txyzs_both,repeat
def get_STD_beaddrift(ims_beads,coord_sel=None,sz_ex=50,desired_nbeads = 20, desired_displ = 0.4, hseed=150,nseed=100,
                      ref=None,force=False,save=True,save_file='temp.pkl'):
    """Given a list of bead images <ims_beads> this handles the fine bead drift correction.
    For each 3d image in <ims_beads> the beads for subimages of size <sz_ex>,
    centered at [center,center,center],[center,center,center]+[0,2*sz_ex,0] are fitted using #get_STD_centers with paramaters <hseed>,<nseed>.
    Beads for each of the two subimages are aligned with the corresponding beads for the reference image of index <ref> (default = len(ims_beads)/2) in ims_beads.
    """
    repeat = True
    txyzs_both = []
    txyzs_both_med = []
    bad_inds = []
    if save:
        save_cor = save_file
        if os.path.exists(save_cor):
            txyzs_both = pickle.load(open(save_cor,'rb'))
            ref_file = os.path.dirname(save_cor)+os.sep+'txyzs_both_med.pkl'
            if os.path.exists(ref_file):
                txyzs_both_med = pickle.load(open(ref_file,'rb'))
                dif_med = 6
                bad_inds = list(np.where([np.sum(np.abs(np.mean(txyz_cur,0)-txyz_med))>dif_med for txyz_cur,txyz_med in zip(txyzs_both,txyzs_both_med)])[0])
            if len(txyzs_both)==len(ims_beads) and not force and len(bad_inds)==0:
                repeat=False
                return txyzs_both,repeat
    if force:
        txyzs_both = []
        bad_inds = []
    if repeat:
        #get txyz
        if ref is None: ref=len(ims_beads)/2
        im_ref = ims_beads[ref]
        coord_sel = np.array(im_ref.shape)/2
        coord_sel1 =coord_sel
        im_ref_sm,coords = grab_block(im_ref,coord_sel1,[sz_ex]*3,return_coords=True)
        cents_ref1 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 1
        coord_sel2 = np.array([0,-2*sz_ex,0])+coord_sel
        im_ref_sm,coords = grab_block(im_ref,coord_sel2,[sz_ex]*3,return_coords=True)
        cents_ref2 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 2
        
        cutoff_=2
        xyz_res_=1
        for iim in bad_inds+range(len(txyzs_both),len(ims_beads)):
            im = ims_beads[iim]
            txy_prev = np.array([0,0,0])
            if iim>0:
                txy_prev = np.mean(txyzs_both[iim-1],axis=0).astype(int)
            if iim<len(txyzs_both_med):
                txy_prev = np.array(txyzs_both_med[iim]).astype(int)
            #print txy_prev,iim,txyzs_both[iim-1]
            im_sm,coords = grab_block(im,coord_sel1+txy_prev,[sz_ex]*3,return_coords=True)
            cents1 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the cube 1
            im_sm,coords = grab_block(im,coord_sel2+txy_prev,[sz_ex]*3,return_coords=True)
            cents2 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the cube 2
            print("Aligning "+str(iim+1))
            #print "Ver 1",len(cents_ref1),len(cents1)
            txyz1,c11,c12 = translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False,return_pts=True)
            #print "Ver 2",len(cents_ref2),len(cents2)
            txyz2,c21,c22 = translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False,return_pts=True)
            #print "Ended"
            txyz = (txyz1+txyz2)/2.
            print(txyz1,txyz2,len(c11),len(c21))
            displ = np.max(np.abs(txyz1-txyz2))
            if (displ>desired_displ) or (len(c11)<desired_nbeads) or (len(c21)<desired_nbeads):
                print("Suspecting failure.")
                #sz_ex+=10
                coord_sel3 = np.array([0,2*sz_ex,0])+coord_sel
                im_ref_sm,coords = grab_block(im_ref,coord_sel3,[sz_ex]*3,return_coords=True)
                cents_ref3 = get_STD_centers(im_ref_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the ref cube 3
                im_sm,coords = grab_block(im,coord_sel3+txy_prev,[sz_ex]*3,return_coords=True)
                cents3 = get_STD_centers(im_sm,hseed=hseed,nseed=nseed)+np.min(coords,axis=-1)#list of fits of beads in the cube 2
                txyz3,c31,c32 = translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=False,return_pts=True)
                measures = list(map(np.sum,list(map(np.abs,[txyz3-txyz1,txyz3-txyz2,txyz1-txyz2]))))
                imeasure = np.argmin(measures)
                nbds = [len(c11),len(c21),len(c31)]
                nbds_variants = [[nbds[2],nbds[0]],[nbds[2],nbds[1]],[nbds[0],nbds[1]]]
                variants = [[txyz3,txyz1],[txyz3,txyz2],[txyz1,txyz2]]
                best_pair = variants[imeasure]
                best_measure = measures[imeasure]
                
                if best_measure>6*desired_displ or np.max(nbds_variants[imeasure])<desired_nbeads:
                    best_pair = [[txyz1,txyz1],[txyz2,txyz2],[txyz3,txyz3]][np.argmax(nbds)]
                
                print(best_pair,measures[imeasure],nbds)
                
                
            else:
                best_pair = [txyz1,txyz2]
            #update txyzs_both
            if iim<len(txyzs_both):
                txyzs_both[iim]= best_pair
            else:
                txyzs_both.append(best_pair)
        if save:
            save_cor = save_file
            pickle.dump(txyzs_both,open(save_cor,'wb'))
    return txyzs_both,repeat
def get_drifts_hindex(drifts_,ih):
    values = []
    for drift_ in drifts_:
        if ih<len(drift_):
            values.append(drift_[ih])
    return np.array(values)
def get_drift_median(drifts_,ih,ignore_bad=True):
    drift_set = get_drifts_hindex(drifts_,ih)
    if ignore_bad:
        keep = np.sum(np.abs(np.squeeze(np.diff(drift_set,axis=1))),axis=-1)<6
    return np.median(np.mean(drift_set,axis=1)[keep],axis=0)
def get_best_drift(drifts_):
    return [get_drift_median(drifts_,i) for i in range(np.max(list(map(len,drifts_))))]
def calc_color_matrix(x,y,order=2):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    y is the Nx3 vector of positions in another channel (i.e. cy7)
    return m_ a 3x7 matrix which when multipled with x,x**2,1 returns y-x
    This m_ is indended to be used with apply_colorcor
    """ 
    x_ = np.array(y)# ref zxy
    y_ = np.array(x)-x_# dif zxy
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    m_ = [np.linalg.lstsq(A, y_[:,iy])[0] for iy in range(len(x_[0]))]
    m_=np.array(m_)
    return m_
def apply_colorcor(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions (typically 750(-->647))
    m is a matrix computed by function calc_color_matrix
    y is the corrected vector in another channel"""
    if m is None:
        return x
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            for j in range(p+1):
                if i+j<=p:
                    exps.append([i,j,p-i-j])
    #find the order
    mx,my = m.shape
    order = int((my-1)/mx)
    assert(my<len(exps))
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy,iz) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy*x_[:,2]**iz)
        A[:,iA]=s
    diff = [np.dot(A,m_) for m_ in m]
    return x_+np.array(diff).T
def sparse_bead_fitting(im_beads,dist_th=7.5,max_n_beads = 3000,plt_val = True):
    """Given an image with beads, this performs the bead fitting forcin sparsit and auto-background thresholding"""
    from skimage.filters import threshold_otsu
    z,x,y,h = get_seed_points_base(im_beads,return_h=True)
    zk,xk,yk,hk = sparse_centers((z,x,y,h),dist_th=dist_th,brightness_th=0,max_num=max_n_beads)
    if plt_val:
        plt.figure()
        plt.hist(np.log(hk),bins=100)
        plt.show()
    h_th = np.exp(threshold_otsu(np.log(hk),nbins=100))
    keep = hk>h_th
    zk,xk,yk,hk = zk[keep],xk[keep],yk[keep],hk[keep]

    pfits = fast_local_fit(im_beads,np.array([zk,xk,yk]).T)
    return pfits
def nan_moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.nancumsum(a_,axis=0, dtype=float)
    ret_nan = ~np.isnan(a_)
    ret_nan = np.cumsum(ret_nan,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_nan[n_:] = ret_nan[n_:] - ret_nan[:-n_]
    ret_ = ret[n_ - 1:] / ret_nan[n_ - 1:]
    return ret_
def moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.cumsum(a_,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_ = ret[n_ - 1:] / n_
    return ret_
def cum_val(vals,target):
    """returns the fraction of elements with value < taget. assumes vals is sorted"""
    niter_max = 10
    niter = 0
    m,M = 0,len(vals)-1
    while True:
        mid = int((m+M)/2)
        if vals[mid]<target:
            m = mid
        else:
            M = mid
        niter+=1
        if (M-m)<2:
            break
    return mid/float(len(vals))
def flatten(l):
    return [item for sublist in l for item in sublist]
def correct_color(dic_cand,htags,dic_chr={},dic_bleed={},
                  zxy_pix_size = [200,109,109]):
    def get_num_cols(tag): 
        num_col = tag.count(',')+1
        if num_col==1:
            num_col=2
        return num_col
    def get_col(tag): 
        return tag.split('_')[-1]
    def get_col_numcols(tag): 
        col = get_col(tag)
        num_col = get_num_cols(tag)
        return col+'_'+str(num_col)+'col'
    cols = np.unique([get_col(tag) for tag in htags])
    num_cols = np.unique([get_num_cols(tag) for tag in htags])
    col_num_cols = np.unique([get_col_numcols(tag) for tag in htags])
    chr_ms = {col:dic_chr.get('m_647_'+col,None) for col in cols}#col_num_cols}
    dic_cand_ = {}
    
    #allhtags = np.unique(list(htags)+list(dic_cand.keys()))
    for htag in htags:
        #col = get_col_numcols(htag)
        col = get_col(htag)
        ps = dic_cand.get(htag,[])
        if len(ps)==0:
            dic_cand_[htag] = {'ps':[],'hs':[],'zxys':[]}
        else:
            zxys = apply_colorcor(ps[:,1:4],chr_ms.get(col,None))*zxy_pix_size
            #print chr_ms.get(col,None) is None
            hs = ps[:,0]
            dic_cand_[htag] = {'ps':ps,'hs':hs,'zxys':zxys}
    if len(dic_bleed)>0:
        #group colors
        htags_gr = partition_map(htags,[htag.split('_')[0]for htag in htags])
        for htags_ in htags_gr:
            for htag1 in htags_:
                for htag2 in htags_:
                    if htag1!=htag2:
                        col1,col2=htag1.split('_')[-1],htag2.split('_')[-1]
                        bleed_m = dic_bleed.get('bleed_'+col1+'_'+col2,None)
                        th_z,th_x,th_y,th_h = dic_bleed.get('thzxyh_'+col1+'_'+col2,[0,0,0,0])
                        ps1 = dic_cand_[htag1]['ps']
                        ps2 = dic_cand_[htag2]['ps']
                        if len(ps1)>0 and len(ps2)>0:
                            hzxys1,hzxys2 = np.array(ps1[:,0:4]),np.array(ps2[:,0:4])
                            hzxys1[:,1:]=apply_colorcor(hzxys1[:,1:],bleed_m)
                            keep = []
                            for h2,z2,x2,y2 in hzxys2:
                                good=True
                                for h1,z1,x1,y1 in hzxys1:
                                    if h1/h2>th_h and np.abs(x1-x2)<th_x and np.abs(y1-y2)<th_y and np.abs(z1-z2)<th_z:
                                        good=False
                                keep.append(good)
                            for key in ['ps','zxys','hs']:
                                dic_cand_[htag2][key] = dic_cand_[htag2][key][keep]
    return dic_cand_
def get_statistical_estimate(dic_cand,htags,Dhs,Ddists,DdistsC,zxys_T=None,nint=5,use_local=True,use_center=True):
    if zxys_T is None:
        zxys_T,hs_T,cols_T = get_maxh_estimate(dic_cand,htags)
    zxys_mv = nan_moving_average(zxys_T,nint)
    zxysC = np.nanmean(zxys_T,axis=0)
    zxys_T = []
    hs_T=[]
    cols_T=[]
    scores_T = []
    all_scores=[]
    for u_i in range(len(htags)):
        htag = htags[u_i]
        col = htag.split('_')[-1]
        cols_T.append(col)
        hs = dic_cand[htag]['hs']
        zxys = dic_cand[htag]['zxys']
       
        if len(zxys)==0:
            zxys_T.append([np.nan]*3)
            hs_T.append(np.nan)
            scores_T.append(np.nan)
            continue
        dists = np.linalg.norm(zxys-zxys_mv[u_i],axis=-1)
        distsC = np.linalg.norm(zxys-zxysC,axis=-1)
        if use_local and use_center:
            scores = [((1-cum_val(DdistsC,dC_))+(1-cum_val(Ddists,d_))+(cum_val(Dhs,h_)))/3 for dC_,d_,h_ in zip(distsC,dists,hs)]
            #scores = [hmean([1-cum_val(DdistsC,dC_),1-cum_val(Ddists,d_),cum_val(Dhs,h_)]) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and use_center:
            scores = [(1-cum_val(DdistsC,dC_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
            #scores = [hmean([1-cum_val(DdistsC,dC_),cum_val(Dhs,h_)]) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if use_local and not use_center:
            scores = [(1-cum_val(Ddists,d_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
            #scores = [hmean([1-cum_val(Ddists,d_),cum_val(Dhs,h_)]) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and not use_center:
            scores = [cum_val(Dhs,h_) for dC_,d_,h_ in zip(distsC,dists,hs)]
        iscore = np.argmax(scores)
        all_scores.append(scores)
        scores_T.append(scores[iscore])
        zxys_T.append(zxys[iscore])
        hs_T.append(hs[iscore])
    zxys_T = np.array(zxys_T)
    hs_T =np.array(hs_T)
    cols_T = np.array(cols_T)
    return zxys_T,hs_T,cols_T,scores_T,all_scores
def get_Ddists_Dhs(zxys_f,hs_f,nint=5):
    h = np.ravel(hs_f)#[np.ravel(cols_f)=='750']
    h = h[np.isnan(h)==False]
    h = np.sort(h)
    dists = []
    distsC = []
    for zxys_T in zxys_f:
        difs = zxys_T-nan_moving_average(zxys_T,nint)#np.nanmedian(zxys_T,0)
        difsC = zxys_T-np.nanmedian(zxys_T,axis=0)
        dists.extend(np.linalg.norm(difs,axis=-1))
        distsC.extend(np.linalg.norm(difsC,axis=-1))
    dists = np.array(dists)
    dists = dists[np.isnan(dists)==False]
    dists = np.sort(dists)
    
    distsC = np.array(distsC)
    distsC = distsC[np.isnan(distsC)==False]
    distsC = np.sort(distsC)
    return h,dists,distsC
def get_maxh_estimate(dic_cand,htags):
    zxys_T = []
    hs_T=[]
    cols_T=[]

    for htag in htags:
        col = htag.split('_')[-1]
        cols_T.append(col)
        hs = dic_cand[htag]['hs']
        zxys = dic_cand[htag]['zxys']
        if len(hs)==0: 
            zxys_T.append([np.nan]*3)
            hs_T.append(np.nan)
            continue
        imax = np.argmax(hs)
        hs_T.append(hs[imax])
        zxys_T.append(zxys[imax])
    return zxys_T,hs_T,cols_T
def get_rough_estimate(dic_cand,chr_pt,htags,chr_ms={},zxy_pix_size = [109,109,200],hybe_start_point = 30,nint = 3,cut_off =np.inf):
    u_i = hybe_start_point
    zxys_T = []
    hs_T=[]
    cols_T=[]
    delta = 1
    center = chr_pt*zxy_pix_size
    while True:
        u_i+=delta
        if u_i>len(htags)-1:
            delta=-1
            u_i=hybe_start_point
            zxys_T = zxys_T[::-1]
            hs_T = hs_T[::-1]
            cols_T = cols_T[::-1]
        if u_i<0:
            break
            

        htag = htags[u_i]
        col = htag.split('_')[-1]
        cols_T.append(col)
        ps = dic_cand.get(htag,[])
        
        if len(ps)==0:
            zxys_T.append([np.nan]*3)
            hs_T.append(np.nan)
            continue
        hs = ps[:,0]

        zxys = apply_colorcor(ps[:,1:4],chr_ms.get(col,None))*zxy_pix_size
        
        if cut_off<np.inf:
            #change center reference
            if len(zxys_T)>nint:
                zxys_int = zxys_T[-nint:]
                center_int = np.nanmean(zxys_int,0)
                if not np.isnan(center_int[0]):
                    center = center_int
                    
            zxys_c = zxys-center
            dists = np.linalg.norm(zxys_c,axis=-1)
            keep = dists<cut_off
            if np.sum(keep)==0: 
                zxys_T.append([np.nan]*3)
                hs_T.append(np.nan)
                continue
            zxys = zxys[keep]
            hs = hs[keep]
        
        zxys_T.append(zxys[np.argmax(hs)])
        hs_T.append(np.max(hs))
    zxys_T = np.array(zxys_T[::-1])
    hs_T =np.array(hs_T[::-1])
    cols_T = np.array(cols_T[::-1])
    return zxys_T,hs_T,cols_T