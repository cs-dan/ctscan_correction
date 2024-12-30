### Dependencies

# Projection & Reconstruction
from AAPMProj import *
from AAPMRecon import *


### Functions

def reconstruction(args1, args2):
    """
    Return sinogram to raw image
    """

    # Set args
    anatomy = args1
    inp_file = 'nonehere'

    # Set FOV params
    if anatomy.lower() == 'h': FOV = 220.16
    else: FOV = 400

    # Reshape
    args2 = args2.reshape(1000, 1, 900)
    
    # Pass into conversion routine
    cfg = AAPMRecon_init(inp_file, FOV)
    raw = recon.recon_alt(cfg, args2)
    
    return raw

def projection(args1, args2):
    anatomy = args1
    inp_file = args2
    if anatomy.lower() == 'h':
        FOV = 220.16
    elif anatomy.lower() == 'o':
        FOV = 400
    else:
        try:
            FOV = float(anatomy)
        except:
            print("Error! Please check the input of anatomy.\n")
    sid = 550.
    sdd = 950.
    
    nrdetcols = 900
    nrcols = 512
    nrrows = 512
    pixsize = FOV/512
    nrviews = 1000
    
    x0 = 0.0/pixsize
    y0 = sid/pixsize
    xCor = 0.0/pixsize
    yCor = 0.0/pixsize
    
    dalpha = 2.*np.arctan2(1.0/2, sdd)
    alphas = (np.arange(nrdetcols)-(nrdetcols-1)/2-1.25)*dalpha
    xds = np.single(sdd*np.sin(alphas)/pixsize)
    yds = np.single((sid - sdd*np.cos(alphas))/pixsize)
    
    viewangles = np.single(1*(0+np.arange(nrviews)/nrviews*2*np.pi))
    
    # Assuming shapes ([1,512,512]) and ([512,512,1]) are 1 to 1 when reshaped into each other
    raw_img = inp_file.reshape([512, 512, 1]).astype(float)
    raw_img = raw_img/1000.*0.02+0.02 # now in the unit of mm^-1
    originalImgPtr = np.single(raw_img)
    sinogram = np.zeros([nrviews, nrdetcols, 1], dtype=np.single) 
        
    sinogram = DD2FanProj(nrdetcols, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)
    sinogram = sinogram*pixsize
    return sinogram
