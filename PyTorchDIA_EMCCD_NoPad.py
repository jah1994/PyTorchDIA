"""PyTorch DIA - Master

**Brief description**

A GPU accelerated approach for fast kernel (and differential background) solutions. The model image proposed in the Bramich (2008) algorithm is analogous to a very simple CNN, with a single convolutional layer / discrete pixel array (i.e. the kernel) and an added scalar bias (i.e. the differential background). We do not solve for the discrete pixel array directly in the linear least-squares sense. Rather, by making use of PyTorch tensors (GPU compatible multi-dimensional matrices) and neural network architecture, we solve via an efficient gradient-descent directed optimisation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import autograd

#from torch.cuda.amp import autocast
#from torch.cuda.amp import GradScaler


print('PyTorch version:', torch.__version__)
# make sure to enable GPU acceleration!
if torch.cuda.is_available() is True:
  device = 'cuda'
else:
  print('CUDA not available, defaulting to CPU')


def convert_to_tensor(image):
  if type(image) is np.ndarray:
      image = image.astype(np.float32)
      image = torch.tensor(image[None, None, :, :])
  else:
      pass

  return image


def infer_kernel(R, I, flat, maxiter, FIM, convergence_plots, d, ks, speedy, tol, lr_kernel, lr_B, SD_steps, Newton_tol):

    '''
    # Arguments
    * 'R' (numpy.ndarray or torch.tensor): The reference image
    * 'I' (numpy.ndarray or torch.tensor): The data/target image
    * 'NM' (numpy.ndarray or torch.tensor): The noise model 'image'
    * 'init_kernel' (torch.tensor): Initial guess for the PSF-matching kernel
    * 'init_B' (torch.tensor): Initial guess for the differential background parameter
    * 'i' (int): Iteration number i.e. M_i estimate

    # Keyword arguments
    * 'maxiter' (int): Maximum number of iterations for the optimisation
    * 'alpha' (float): Strength of the L2 regularisation penalty
    * 'FIM' (bool): Calculate parameter uncertanties from the Fisher Matrix
    * 'convergence_plots' (bool): Plot parameter estimates vs steps after optimising
    * 'd' (int): Polynomial of degree 'd' for fitting a spatially varying background
    * 'ks' (int): kernel of size ks x ks, needs to be odd
    * speedy (bool): If True, don't bother with linear transformation operation
      i.e. to be used if we're solving for a scalar background parameter
    * tol (float): Minimum relative change in parameter values before claiming convergence
    * lr_kernel (float): Steepest descent learning rate for kernel
    * lr_B (float): Steepest descent learning rate for background parameter(s)
    * SD_steps (int): Number of gradient descent steps to talke before switching to quasi-Newton optimisation
    
    # returns
    * 'kernel' (numpy.ndarray): the (flipped) inferred kernel
    * 'inferred_bkg' (float): B_0 background term
    * 'fit' (numpy.ndarray): the spatially varying background (inferred_bkg == fit if d=0)
    '''
    
    R, I = convert_to_tensor(R), convert_to_tensor(I)
    
    if speedy is True:
      model = torch.nn.Sequential(
          torch.nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=ks,
                          padding = 0,
                          bias=True

        )
      )

      '''
      # Initialise kernel and bias
      model[0].weight = torch.nn.Parameter(1e-3*torch.ones(model[0].weight.shape, requires_grad=True))
      model[0].bias = torch.nn.Parameter(1e3*torch.ones(model[0].bias.shape, requires_grad=True))
      '''
      # Initialise kernel and bias
      init_kernel_pixels = 1. / (ks**2) # ensures that the kernel sums to 1 at initialisation
      init_background = torch.median(I).item() # estimate for the 'sky' level of the target image
      model[0].weight = torch.nn.Parameter(init_kernel_pixels*torch.ones(model[0].weight.shape, requires_grad=True))
      model[0].bias = torch.nn.Parameter(init_background*torch.ones(model[0].bias.shape, requires_grad=True))
     
    
    else:
      class model(torch.nn.Module):
          def __init__(self):

              super(model, self).__init__()
              self.conv = torch.nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=ks,
                              padding = np.int((ks/2)-0.5),
                              padding_mode = 'zeros',
                              bias=False)
              
              self.poly = torch.nn.Linear(2*d+1, 1, bias=False)
              #self.poly = torch.nn.Linear(500**2, 1, bias=False)
           
          def forward(self, x, A):
              reshaped_size = (1, 1, R[0][0][0].size()[0], R[0][0][0].size()[0])
              y_pred =  torch.add(self.conv(x), torch.reshape(self.poly(A), reshaped_size))
              return y_pred
    
      model = model()

      # Construct the design/weight matrix for the polynomial background fit
      x = np.linspace(-0.5, 0.5, R[0][0][0].size()[0])
      y = np.linspace(-0.5, 0.5, R[0][0][0].size()[0])
      X, Y = np.meshgrid(x, y, copy=False)
      X = X.flatten()
      Y = Y.flatten()
      
      if d == 0:
        A = np.array([X*0+1]).T
        def poly2Dreco(X, Y, c):
          return c[0]
      elif d == 1:
        A = np.array([X*0+1, X, Y]).T      
        def poly2Dreco(X, Y, c):
          return c[0] + X*c[1] + Y*c[2]
      elif d == 2:
        A = np.array([X*0+1, X, Y, X**2, Y**2]).T
        def poly2Dreco(X, Y, c):
          return c[0] + X*c[1] + Y*c[2] + X**2*c[3] + Y**2*c[4]
      elif d == 3:
        A = np.array([X*0+1, X, Y, X**2, Y**2, X**3, Y**3]).T
        def poly2Dreco(X, Y, c):
          return c[0] + X*c[1] + Y*c[2] + X**2*c[3] + Y**2*c[4] + X**3*c[5] + Y**3*c[6]
      else:
        print('Polynomial of d=%d not currenty supported... need to automate this, reverting to d=3' % d)
        A = np.array([X*0+1, X, Y, X**2, Y**2, X**3, Y**3]).T
        def poly2Dreco(X, Y, c):
          return c[0] + X*c[1] + Y*c[2] + X**2*c[3] + Y**2*c[4] + X**3*c[5] + Y**3*c[6]
      
      if torch.cuda.is_available() is True:
        A = torch.tensor(A).to(device).float()
      else:
        A = torch.tensor(A).float()


      model.conv.weight = torch.nn.Parameter(1e-3* torch.ones(model.conv.weight.shape, requires_grad=True))
      model.poly.weight = torch.nn.Parameter(1* torch.ones(model.poly.weight.shape, requires_grad=True))

    # Move model to GPU
    if torch.cuda.is_available() is True:
      model = model.to(device)

    # Define the loss (our scalar objective function to optimise)
    # Sometimes referred to as the 'cost' in the ML literature
    class negative_log_likelihood(torch.nn.Module):
    
        def forward(model, targ, flat, c):
            #var = model + sigma0**2
            #chi2 = torch.sum((model - targ) ** 2 / var)
            #ln_sigma = torch.sum(torch.log(var))
            #nll = 0.5 * (chi2 + ln_sigma)

            # Total gain, G, and EMCCD excess noise factor, E
            G = 25.8/300.
            E = 2

            shot_noise = torch.clamp(model, min=1.)/(G*flat)
            var = E*shot_noise

            NM = torch.sqrt(var)
            ln_sigma = torch.sum(torch.log(NM))
           
            # gaussian when (model - targ)/NM <= c
            # absolute deviation when (model - targ)/NM > c
            cond1 = torch.abs((model - targ)/NM) <= c
            cond2 = torch.abs((model - targ)/NM) > c
            inliers = ((model - targ)/NM)[cond1]
            outliers = ((model - targ)/NM)[cond2]

            l2 = 0.5*torch.sum(torch.pow(inliers, 2))
            l1 = (c * torch.sum(torch.abs(outliers)) - (0.5 * c**2))

            nll = l2 + l1 + ln_sigma

            return nll

    # Keep track of the speed to convergence for development's sake
    losses = []
    ts = []

    # prepare optimizers - For (steepest) gradient descent, we use Adam
    # and once we get close to the minimum, we switch to L-BFGS
    if speedy is True:

      optimizer_Adam = torch.optim.Adam([
                      {'params': model[0].weight, 'lr': lr_kernel},
                      {'params': model[0].bias, 'lr': lr_B}
                  ])
                  
    else:

        optimizer_Adam = torch.optim.Adam([
                    {'params': model.conv.weight, 'lr': lr_kernel},
                    {'params': model.poly.weight, 'lr': lr_B}
                ])
                
    
    optimizer_Newton = torch.optim.LBFGS(model.parameters(), tolerance_change=tol, history_size=10, line_search_fn=None)
    #optimizer_Newton = torch.optim.LBFGS(model.parameters(), tolerance_change=tol, history_size=100, line_search_fn='strong_wolfe')

    # L-BFGS needs to evaulte the scalar objective function multiple times each call, and requires a
    # closure to be fed to opitmizer_Newton
    def closure():
      optimizer_Newton.zero_grad()
      y_pred = model(R)
      loss = negative_log_likelihood.forward(y_pred, I, flat, c)
      loss.backward()
      return loss

        
    # Time the optimisation
    start_time_infer = time.time()

    # flag to switch to quasi newton step
    use_Newton = False
    

    torch.set_printoptions(precision=10)
    print('Check dtype of data and weights:')
    print(R.dtype, I.dtype, model[0].weight.dtype, model[0].bias.dtype)
    print('Check size of data and weights:')
    print(R.size(), I.size(), model[0].weight.size(), model[0].bias.size())
    
    ## scaled gradients ###
    #scaler = GradScaler()

    ## begin optimising ##
    print('Starting optimisation')
    for t in range(maxiter):

        #if t < 150:
        #  c = 10
        #else:
        #  c = 1.345
        c = 1.345


        # flag to use steepest decent if relative change in loss
        # not below Newton_tol
        if use_Newton == False:

          '''

          optimizer_Adam.zero_grad()

          
          # Runs the forward pass with autocasting
          with autocast():

            if speedy is True:
              y_pred = model(R)
            else:
              y_pred = model(R, A)
      
          # compute the loss
          loss = negative_log_likelihood.forward(y_pred, I)

          scaler.scale(loss).backward()
          scaler.step(optimizer_Adam)
          # Updates the scale for next iteration
          scaler.update()
          '''
          if speedy is True:
            y_pred = model(R)
          else:
            y_pred = model(R, A)
      
          # compute the loss
          loss = negative_log_likelihood.forward(y_pred, I, flat, c)
         
          # clear gradients, compute gradients, take a single
          # steepest descent step
          optimizer_Adam.zero_grad()
          loss.backward()
          optimizer_Adam.step()
          
          # append loss
          losses.append(loss.detach())
          ts.append(t)
        
        # don't take more than 1000 Newton steps
        elif use_Newton == True and t < SD_steps_taken + 1000:
          # perform a single optimisation (quasi-Newton) step
          optimizer_Newton.step(closure)

          # compute and append new loss after the update
          # must be a way to improve this.... #
          y_pred = model(R)
          loss = negative_log_likelihood.forward(y_pred, I, flat, c)
          losses.append(loss.detach())
          ts.append(t)
          
        else:
          print('Failed to converge!')
          break


        # Convergence reached if less than specified tol and more than 100
        # steps taken (guard against early stopping)
        if speedy is True:
          if t>300 and abs((losses[-1] - losses[-2])/losses[-2]) < tol:
            print('Converged!')
            print('Total steps taken:', t)
            try:
              print('SD steps:', SD_steps_taken)
              print('L-BFGS steps:', t - SD_steps_taken)
            except UnboundLocalError:
                print('SD only')
            break

          elif t>250 and abs((losses[-1] - losses[-2])/losses[-2]) < Newton_tol and use_Newton == False:
            use_Newton = True
            SD_steps_taken = t
            print('Switching to Quasi-Newton step after %d SD steps' % SD_steps_taken)
          elif t == maxiter - 1:
            print('Failed to converge!')
            break
      

    print("--- Finished kernel and background fit in %s seconds ---" % (time.time() - start_time_infer))


    if speedy is True:
      kernel, B = model[0].weight, model[0].bias

    else:
      kernel, B = model.conv.weight, model.poly.weight


    def compute_full_hessian(grads):

      #Note the use of .detach(). In general, computations involving
      #variables that require gradients will keep history.

      grads = torch.cat((grads[0].flatten(), grads[1].flatten()))
      grad = grads.reshape(-1)
      d = len(grad)
      H = torch.zeros((d, d))
      
      t = time.time()
      print('Looping...')
      for i,dl_dthetai in enumerate(grads):
        H_rowi = torch.autograd.grad(dl_dthetai, model.parameters(), retain_graph=True)
        H_rowi = torch.cat((H_rowi[0].flatten(), H_rowi[1].flatten()))
        H[i] = H_rowi.detach()
      print('Looping took %s seconds' % (time.time() - t))
      return H


    def compute_hessian_blockdiags(grads, params):
      H = []
      t = time.time()
      print('Looping...')
      for i, (grad, p) in enumerate(zip(grads, params)):
          grad = grad.reshape(-1)
          d = len(grad)
          dg = torch.zeros((d, d))

          for j, g in enumerate(grad):
              g2 = autograd.grad(g, p, create_graph=True)[0].view(-1)
              dg[j] = g2.detach()

          H.append(dg)
      print('Looping took %s seconds' % (time.time() - t))
      return H

    
    if FIM == True:
      '''
      We're minimizing an approximation to the negative log-likelihood
      So the returned Hessian is equivalent to the observed Fisher
      Information Matrix (i.e. FIM evaluated at MLE)
      '''

      def det_test(matrix):
        sign, logdet = torch.slogdet(matrix)
        if sign.item() <= 0.:
          print('Covariance matrix not positive definite! Sign of determinant:', sign.item())
        elif sign.item() > 0.:
          pass

      def eigenvals_test(matrix):
        eigenvals = torch.eig(matrix)[0]
        if any(eigval <= 0. for eigval in eigenvals[:,0]):
          print('Covariance matrix not positive definite!. Non-positive eigenvalues.1')

      
      def get_stderror(obs_fisher_matrix):
        '''
        Is the estiamted covariance matrix valid?
        A valid covariance matrix has to be positive definite
        Test 1:
        check if det cov_matrix <= 0., cov_matrix is not valid
        Test 2:
        diagnoalise cov_matrix to determine eigenvalues.
        If any of these are <= 0., cov_matrix is not valid
        '''
        cov_matrix = torch.inverse(obs_fisher_matrix)
        print('Covariance Matrix:', cov_matrix)
        det_test(cov_matrix) # Test 1
        eigenvals_test(cov_matrix) # Test 2
        cov_matrix_diagonals = torch.diag(cov_matrix)

        return cov_matrix_diagonals
      '''
      Compute FIM (i.e. Hessian!)
      '''
      
      # Full Hessian #
      if speedy is True:
        y_pred = model(R)
      else:
        y_pred = model(R, A)
      loss = negative_log_likelihood.forward(y_pred, I, flat, c)
      logloss_grads = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
      print('Building full Hessian...')
      full_hessian_time = time.time() 
      H = compute_full_hessian(logloss_grads)

      print('---Finished constructing full hessian in %s seconds---' % (time.time() - full_hessian_time))      
      cov_matrix_diags = get_stderror(H)
      print('Cov matrix diagonals:', cov_matrix_diags)
      psf_err, B0_err = torch.sqrt(torch.sum(cov_matrix_diags[0:-(2*d+1)])), torch.sqrt(cov_matrix_diags[-(2*d+1)])
      print('Photometric scaling:', torch.sum(kernel).item(), '+/-', psf_err.item())
      if speedy is True:
        print('B_0:', torch.sum(B[0]).item(), '+/-', B0_err.item())
      else:
        print('B_0:', torch.sum(B[0][0]).item(), '+/-', B0_err.item())


    else:
      print('Photometric scaling:', torch.sum(kernel))
      if speedy is True:
        print('B_0:', torch.sum(B[0]).item())
      else:
        print('B_0:', torch.sum(B[0][0]).item())
 
    if convergence_plots == True:
      plt.plot(ts[1:], np.log(losses[1:]))
      plt.xlabel('Iterations')
      plt.ylabel('log_10(loss)')
      plt.title('log loss vs iterations')
      plt.show()
    

    # flip kernel to correct orientation (as I pass this to conv2d)
    kernel = torch.flip(kernel, [2, 3])
    kernel = kernel[0][0].cpu().detach().numpy()
    B = B[0].cpu().detach().numpy()

    if speedy is True:
      B = B.item()

    if d>0:
      X, Y = np.meshgrid(x, y, copy=False)
      B = poly2Dreco(X, Y, B)
    
    return kernel, B



def DIA(R,
        I,
        flat,
        read_noise = 0.,
        ks = 15,
        lr_kernel = 1e-2,
        lr_B = 1e1,
        SD_steps = 100,
        Newton_tol = 1e-6,
        poly_degree=0,
        fast=True,
        tol = 1e-5,
        max_iterations = 5000,
        fisher=False,
        show_convergence_plots=False):
  
  '''
  ## Arguments
  * 'R_full' (numpy.ndarray): Input reference image
  * 'I_full'(numpy.ndarray): Input target image
  * 'flat' (numpy.ndarray): provided flat field for the images

  ## Keyword Arguments
  * 'read_noise' (float): detector read noise (ADU), defeault = 0.
  * 'unweighted' (bool): don't bother with a noise model, default=False
  * 'n_samples' (int): If MC sampling, specify how many kernel and background solutions we want, default = 1
  * 'full_image' (bool): Infer kernel for full image or MC sampled subregions, default=True
  * 'display_stamps' (bool): Plot the input reference and target pair (either full image or subregions), default=False
  * 'sky_subtract' (bool): Subtract the median pixel values from the input images, default=False
  * 'iters' (int): Number of iterations (estimates of Model image) to perform, default = 3
  * 'ks' (int): Size of ks x ks kernel **Needs to be odd**, default=15
  * 'lr_kernel' (float): The learning rate for the parameters of the convolution kernel, default=1e-2
  * 'lr_B' (float): The learning rate for the parameters for the differential background solution, default=1e1
  * 'SD_steps' (int): Number of gradient descent steps to talke before switching to quasi-Newton optimisation
  * 'poly_degree' (int): Degree of polynomial for background fit, default=0
  * 'fast' (bool): Use in-built torch.nn.Conv2d function if True, which fits for a scalar background term.
     If False, we'll fit a polynomial of degree 'poly_degree' for the background, which requires an additional
     customised linear transformation operation, which is slower than the in-built function even for 0 degree
     polynomial, default = True.
  * 'tol' (float): Minimum relative change in parameters for claiming convergence of kernel and background fit,
     default = 1e-5
  * 'alpha' (float): Strength of L2 regularisation penalty on kernel solution, default = 0.
  * 'max_iterations' (int): Maximum number of iterations in optimisation, default=5000
  * 'fisher' (bool): Output kernel and background uncertainty estimates calculated from Fisher Matrix, default=False
  * 'show_convergence_plots' (bool): Plot parameter estimates vs steps in optimisation procedure, default=False
  * 'display_D' (bool): Plot the difference image(s), default = False
  * 'k' (int): sigma clip to apply to normalised residuals at each iteration,
     default=5 **WARNING:Highly sensitve to choice of noise model!!
     If you're not sure, set this to be very large to avoid overly severe clipping!**
  * 'precision' (int): Decimal place precision at which we claim convergence of kernel solution,
     default=3 i.e. if change in photometric scale factor at next iteration < 0.001 we've converged
  * 'display_masked_stamps' (bool): Plot the sigma clipped images, default=False
  * 'display_M' (bool): Plot the model image(s)
  * 'display_kernel' (bool): Plot the inferred kernel(s)
  * 'display_B* (bool): Plot the inferred spatially varying background(s)
  
  ## Returns
  * 'kernel' (numpy.ndarray): the convolution kernel
  * 'B' (numpy.ndarray): the differential background
  '''


  start_time_total = time.time()
  
  # trim I such that target image pixels correspond to only those with valid convolution computations
  hwidth = np.int((ks - 1) / 2)
  nx, ny = I.shape
  I = I[hwidth:nx-hwidth, hwidth:nx-hwidth]
  flat = flat[hwidth:nx-hwidth, hwidth:nx-hwidth]
    
  #### Convert numpy images to tensors and move to GPU
  I, R, flat = convert_to_tensor(I), convert_to_tensor(R), convert_to_tensor(flat)

  time_to_move_to_GPU = time.time()
    
  # Move to GPU if CUDA available
  if torch.cuda.is_available() is True:
    R = R.to(device)
    I = I.to(device)
    flat = flat.to(device)
   


  print("--- Time to move data onto GPU: %s ---" % (time.time() - time_to_move_to_GPU))



  kernel, B = infer_kernel(R, I, flat, 
				   maxiter=max_iterations,
				   FIM=fisher,
				   convergence_plots=show_convergence_plots,
				   d = poly_degree,
				   ks = ks,
				   speedy = fast,
				   tol = tol,
				   lr_kernel = lr_kernel,
				   lr_B = lr_B,
				   SD_steps = SD_steps,
				   Newton_tol = Newton_tol)


  print("--- Finished in a total of %s seconds ---" % (time.time() - start_time_total))

  return kernel, B






