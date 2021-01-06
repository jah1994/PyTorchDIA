import numpy as np

def make_noiseless_image(shape, positions, fluxes, sky, psf_sigma):
    """
    Make one image.
    """
    fluxes_in_stamp = []
    img = np.zeros(shape)
    img += sky
    nx, ny = shape
    xg, yg = np.meshgrid(range(nx), range(ny))
    for pos, f in zip(positions, fluxes):
        #print(pos[0], pos[1])
        kernel = np.exp(-0.5 * ((xg - pos[0]) ** 2 + (yg - pos[1]) ** 2)
                / psf_sigma ** 2)
                
        #img += f * kernel / (2. * np.pi * psf_sigma ** 2) ## original line  - doesn't guarantee kernel normalised ###
        kernel /= np.sum(kernel) # normalise kernel
        #print('kernel sum:', np.sum(kernel))
        img += f * kernel

        #if 19 < pos[0] < img.shape[0] - 19 and 19 < pos[1] < img.shape[1] - 19:
        #    fluxes_in_stamp.append(f)
        #img += f*(kernel/np.sum(kernel))
        
        #print('f', f)
        #print('kernel sum', np.sum(kernel))
        #print('sum', np.sum(img))
        #stop = input()
        
    return img, fluxes_in_stamp

def MakeFake(N, size, n_sources, psf_sigma, sky, positions, fluxes, shifts):
    '''
    N: Number of images
    size: image axis length (assumed to be square)
    n_sources: Number of sources
    psf_sigma: std of gaussian psf profile
    sky: sky level (ADU)
    '''

    #np.random.seed()
    shape = (size, size)

    # positions
    #positions_x = np.random.uniform(0, size, (n_sources,1))
    #positions_y = np.random.uniform(0, size, (n_sources,1))
    #positions = np.hstack((positions_x, positions_y))

    # fluxes
    #F = np.random.uniform(10**(-9), 10**(-4.5), n_sources)
    #fluxes = F**(-2./3.)

    ## relocate the brightest source to the image centre
    centre = np.int(size/2)
    F_max_index = np.where(fluxes == np.max(fluxes))
    #print(shifts[0], shifts[1])
    positions[F_max_index] = np.array([[centre + shifts[0], centre + shifts[1]]])
    #F_min_index = np.where(fluxes == np.min(fluxes))
    #positions[F_min_index] = np.array([[centre, centre]])

    # F_max / total flux in image (the 104 x 104 region used for the inference)
    print('Max flux:', np.max(fluxes))
    print('Frac for 142x142 image:', np.max(fluxes) / np.sum(fluxes))

    for n in range(N):
        img, fluxes_in_stamp = make_noiseless_image(shape, positions, fluxes, sky, psf_sigma)
        #F_frac = np.max(fluxes_in_stamp) / np.sum(fluxes_in_stamp)
        F_frac = np.max(fluxes) / np.sum(fluxes)
        return img, F_frac
