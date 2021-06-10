"""
** PyTorch DIA **

A GPU accelerated approach for fast kernel (and differential background) solutions. The model image proposed in the Bramich (2008) algorithm is analogous to a very simple CNN, with a single convolutional layer / discrete pixel array (i.e. the kernel) and an added scalar bias (i.e. the differential background). We do not solve for the discrete pixel array directly in the linear least-squares sense. Rather, by making use of PyTorch tensors (GPU compatible multi-dimensional matrices) and neural network architecture, we solve via an efficient gradient-descent directed optimisation.
"""

## imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import autograd

# what version are we running?
print('PyTorch version:', torch.__version__)

# make sure to enable GPU acceleration (if availabel)!
if torch.cuda.is_available() is True:
  device = 'cuda'
else:
  print('CUDA not available, defaulting to CPU')


# PyTorch Conv2D requires 4D tensors
def convert_to_tensor(image):
  if type(image) is np.ndarray:
      image = image.astype(np.float32)
      image = torch.tensor(image[None, None, :, :])
  else:
      pass

  return image


def infer_kernel(R, I, flat, loss_fn, maxiter, FIM, alpha, convergence_plots, d, ks, tol, lr_kernel, lr_B, Newton_tol):

    '''
    # Arguments
    * 'R' (numpy.ndarray or torch.tensor): The reference image
    * 'I' (numpy.ndarray or torch.tensor): The data/target image

    # Keyword arguments - See DIA()
    
    # returns
    * 'kernel' (numpy.ndarray): the (flipped) inferred kernel
    * 'inferred_bkg' (float): B_0 background term
    * 'fit' (numpy.ndarray): the spatially varying background (inferred_bkg == fit if d=0)
    '''
    
    # Initialise model parameters
    init_kernel_pixels = 1. / (ks**2) # ensures that the kernel sums to 1 at initialisation
    init_background = torch.median(I).item() # estimate for the 'sky' level of the target image
    
    if d == 0:
      model = torch.nn.Sequential(
          torch.nn.Conv2d(in_channels=1,
                          out_channels=1,
                          kernel_size=ks,
                          padding = 0,
                          bias=True

        )
      )


      model[0].weight = torch.nn.Parameter(init_kernel_pixels * torch.ones(model[0].weight.shape, requires_grad=True))
      model[0].bias = torch.nn.Parameter(init_background * torch.ones(model[0].bias.shape, requires_grad=True))

     
    
    else:
    
      # TODO: automate polynomial design matrix construction
      
      class model(torch.nn.Module):
          def __init__(self):

              super(model, self).__init__()
              self.conv = torch.nn.Conv2d(in_channels = 1,
                              out_channels = 1,
                              kernel_size = ks,
                              padding = 0,
                              bias = False)
              
              self.poly = torch.nn.Linear(2*d+1, 1, bias=False)
           
          def forward(self, x, A):
              reshaped_size = (1, 1, I[0][0][0].size()[0], I[0][0][0].size()[0])
              y_pred =  torch.add(self.conv(x), torch.reshape(self.poly(A), reshaped_size))
              return y_pred
    
      model = model()

      # Construct the design/weight matrix for the polynomial background fit
      x = np.linspace(-0.5, 0.5, I[0][0][0].size()[0])
      y = np.linspace(-0.5, 0.5, I[0][0][0].size()[0])
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


      model.conv.weight = torch.nn.Parameter(init_kernel_pixels * torch.ones(model.conv.weight.shape, requires_grad=True))
      bkg_init = torch.ones(model.poly.weight.shape)
      bkg_init[0][0] = torch.median(I).item()
      model.poly.weight = torch.nn.Parameter(bkg_init, requires_grad=True)

    # Move model to GPU
    if torch.cuda.is_available() is True:
      model = model.to(device)

    # And if alpha != 0, construct the Laplacian to add as a prior
    # to penalize the log-likelihood
    if alpha != 0.:
    
        Nk = model[0].weight[0][0].flatten().size()[0]# number of DBFs
        print('Constructing Laplacian for square kernel with %d DBFs' % Nk)
        
        L = np.zeros((Nk, Nk))
        
        centre_bit = np.arange(1, np.sqrt(Nk)-1)
        
        # corners
        L[0][0] = 2
        L[-1][-1] = 2
        L[np.int(np.sqrt(Nk)-1)][np.int(np.sqrt(Nk)-1)] = 2
        L[Nk-np.int(np.sqrt(Nk))][Nk-np.int(np.sqrt(Nk))] = 2
        
        for u in range(0, Nk):
            for v in range(0, Nk):
                # diagonals
                if u == v:
                    # centre bit
                    if np.sqrt(Nk) < u < Nk - np.sqrt(Nk) - 1 and u % np.sqrt(Nk) in centre_bit:
                        L[u][v] = 4
                    
                    # rest of diags are 3
                    elif L[u][v] != 2:
                        L[u][v] = 3
                
                # off-diagonals        
                elif u != v:
                    if u == v+1:
                        if (v+1) % np.sqrt(Nk) != 0:
                            L[u][v] = -1
                            L[v][u] = -1
                            
                    elif u == v + np.sqrt(Nk):
                        L[u][v] = -1
                        L[v][u] = -1
    
    
        # convert L to tensor
        L = torch.from_numpy(L).float()
        # move L to cuda
        if torch.cuda.is_available() is True:
            L = L.to(device)
            
    # target image pixels
    N_dat = I[0][0].flatten().size()[0]
    
    # adds laplacian prior on kernel pixels            
    def add_Laplacian_prior(w, loss):
        vector = w[0][0].flatten()
        prior = alpha * N_dat * (vector.t() @ L.t() @ L @ vector)
        loss += prior
        return loss


    # Keep track of the progress of the optimisation
    losses = []
    ts = []

    # initialise optimizers - For (steepest) gradient descent, we use Adam
    # and once we get close to the minimum, we switch to L-BFGS
    if d == 0:

      optimizer_Adam = torch.optim.Adam([
                      {'params': model[0].weight, 'lr': lr_kernel},
                      {'params': model[0].bias, 'lr': lr_B}
                  ])
                  
    else:

        optimizer_Adam = torch.optim.Adam([
                    {'params': model.conv.weight, 'lr': lr_kernel},
                    {'params': model.poly.weight, 'lr': lr_B}
                ])
                
 
    # flag which controls when to switch from Adam to L-BFGS
    use_Newton = False   
    optimizer_Newton = torch.optim.LBFGS(model.parameters(), tolerance_change=tol, history_size=10, line_search_fn=None)

    # L-BFGS needs to evaulte the scalar objective function multiple times each call, and requires a
    # closure to be fed to opitmizer_Newton
    def closure():
      optimizer_Newton.zero_grad()
      if d == 0:
        y_pred = model(R)
      else:
        y_pred = model(R, A)      
       
      loss = loss_fn(y_pred, I, flat)
      
      if alpha != 0:
        try:
            loss = add_Laplacian_prior(model[0].weight, loss)
        except TypeError:
            loss = add_Laplacian_prior(model.conv.weight, loss)
      
      loss.backward()
      return loss

    
    torch.set_printoptions(precision=10)
    try:
        print('Check dtype of data and weights:')
        print(R.dtype, I.dtype, model[0].weight.dtype, model[0].bias.dtype)
        print('Check size of data and weights:')
        print(R.size(), I.size(), model[0].weight.size(), model[0].bias.size())
    except TypeError:
        print('Check dtype of data and weights:')
        print(R.dtype, I.dtype, model.conv.weight.dtype, model.poly.weight.dtype)
        print('Check size of data and weights:')
        print(R.size(), I.size(), model.conv.weight.size(),  model.poly.weight.size())
        
    ## begin optimising ##
    print('Starting optimisation')
    
    # Time the optimisation
    start_time_infer = time.time()
    
    for t in range(maxiter):

        # flag to use steepest decent if relative change in loss
        # not below Newton_tol
        
        if use_Newton == False:

          if d == 0:
            y_pred = model(R)
          else:
            y_pred = model(R, A)
      
          # compute the loss
          loss = loss_fn(y_pred, I, flat)
          
          if alpha != 0:
            try:
                loss = add_Laplacian_prior(model[0].weight, loss)
            except TypeError:
                loss = add_Laplacian_prior(model.conv.weight, loss)
           
         
          # clear gradients, compute gradients, take a single
          # steepest descent step
          optimizer_Adam.zero_grad()
          loss.backward()
          optimizer_Adam.step()
          
          # append loss
          losses.append(loss.detach())
          ts.append(t)
        
        # don't take more than 250 Newton steps
        elif use_Newton == True and t < SD_steps_taken + 250:
        
          # perform a single optimisation (quasi-Newton) step
          optimizer_Newton.step(closure)

          # compute and append new loss after the update
          # must be a way to improve this.... #
          if d == 0:
            y_pred = model(R)
          else:
            y_pred = model(R, A)
      
          loss = loss_fn(y_pred, I, flat)
          
          if alpha != 0:
            try:
                loss = add_Laplacian_prior(model[0].weight, loss)
            except TypeError:
                loss = add_Laplacian_prior(model.conv.weight, loss)
           
         
          losses.append(loss.detach())
          ts.append(t)
          
        else:
          print('Failed to converge!')
          break


        # Convergence reached if less than specified tol and more than 100
        # steps taken (guard against early stopping)
        if t>100 and abs((losses[-1] - losses[-2])/losses[-2]) < tol:
          print('Converged!')
          print('Total steps taken:', t)
          try:
            print('SD steps:', SD_steps_taken)
            print('L-BFGS steps:', t - SD_steps_taken)
          except UnboundLocalError:
              print('SD only')
          break

        elif t>100 and abs((losses[-1] - losses[-2])/losses[-2]) < Newton_tol and use_Newton == False:
          use_Newton = True
          SD_steps_taken = t
          print('Switching to Quasi-Newton step after %d SD steps' % SD_steps_taken)
        elif t == maxiter - 1:
          print('Failed to converge!')
          break
      

    print("--- Finished kernel and background fit in %s seconds ---" % (time.time() - start_time_infer))


    if d == 0:
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
        det_test(cov_matrix) # Test 1
        eigenvals_test(cov_matrix) # Test 2
        cov_matrix_diagonals = torch.diag(cov_matrix)

        return cov_matrix_diagonals
      '''
      Compute FIM (i.e. Hessian!)
      '''
      
      # Full Hessian #
      if d == 0:
        y_pred = model(R)
      else:
        y_pred = model(R, A)
      loss = loss_fn(y_pred, I, flat)
      
      if alpha != 0:
        try:
            loss = add_Laplacian_prior(model[0].weight, loss)
        except TypeError:
            loss = add_Laplacian_prior(model.conv.weight, loss)
            
      logloss_grads = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
      print('Building full Hessian...')
      full_hessian_time = time.time() 
      H = compute_full_hessian(logloss_grads)

      print('---Finished constructing full hessian in %s seconds---' % (time.time() - full_hessian_time))      
      cov_matrix_diags = get_stderror(H)
      print('Cov matrix diagonals:', cov_matrix_diags)
      psf_err, B0_err = torch.sqrt(torch.sum(cov_matrix_diags[0:-(2*d+1)])), torch.sqrt(cov_matrix_diags[-(2*d+1)])
      print('Photometric scaling:', torch.sum(kernel).item(), '+/-', psf_err.item())
      if d == 0:
        print('B_0:', torch.sum(B[0]).item(), '+/-', B0_err.item())

    else:
    
      print('Photometric scaling:', torch.sum(kernel).item())
      if d == 0:
        print('B_0:', torch.sum(B[0]).item())
      else:
        print('B coeffs:', B[0].cpu().detach())
 
    if convergence_plots == True:
      plt.plot(ts[1:], np.log(losses[1:]))
      plt.grid()
      plt.xlabel('Iterations')
      plt.ylabel('log_10(loss)')
      plt.title('log loss vs iterations')
      plt.show();
    

    # flip kernel to correct orientation (as I pass this to conv2d)
    kernel = torch.flip(kernel, [2, 3])
    kernel = kernel[0][0].cpu().detach().numpy()
    B = B[0].cpu().detach().numpy()

    if d == 0:
      B = B.item()
    
    return kernel, B, y_pred[0][0].detach().cpu().numpy()



def DIA(R,
        I,
        loss_fn,
        flat = 1,
        ks = 19,
        lr_kernel = 1e-3,
        lr_B = 1,
        max_iterations = 10000,
        poly_degree = 0,
        alpha = 0,
        Newton_tol = 1e-6,
        tol = 1e-9,
        fisher = False,
        show_convergence_plots = False):
  
  '''
  ## Arguments
  * 'R_full' (numpy.ndarray): Input reference image
  * 'I_full'(numpy.ndarray): Input target image
  * 'loss_fn' (Pythonic function composed of torch.Tensors): The scalar objective function to minimise

  ## Keyword Arguments
  * 'ks' (int): Size of ks x ks kernel **Must be odd**
  * 'lr_kernel' (float): The learning rate for the parameters of the convolution kernel, default=1e-3
  * 'lr_B' (float): The learning rate for the parameters for the differential background solution, default=1
  * 'max_iterations' (int): Maximum Number of optimisation steps
  * 'poly_degree' (int): Degree of polynomial for background fit, default=0
  * 'alpha' (float): Strength of LaPlacian smoothing on kernel, default = 0.
  * 'Newton_tol' (float): tol at which to switch from steepest gradient descent to L-BFGS
  * 'tol' (float): Minimum relative change in parameters for claiming convergence of kernel and background fit,
     default = 1e-9
  * 'fisher' (bool): Output kernel and background uncertainty estimates calculated from Fisher Matrix, default=False WARNING: THIS HAS NOT BEEN TESTED FOR CURRENT CODE VERSION
  * 'show_convergence_plots' (bool): Plot Loss vs steps in optimisation procedure, default=False
  
  ## Returns
  * 'kernel' (numpy.ndarray): the convolution kernel
  * 'B' (numpy.ndarray): the differential background
  * 'model' (numpy.ndarray): the DIA image model
  '''

  start_time_total = time.time()
  
  # trim I such that target image pixels correspond to only those with valid convolution computations
  hwidth = np.int((ks - 1) / 2)
  nx, ny = I.shape
  I = I[hwidth:nx-hwidth, hwidth:nx-hwidth]

  #### Convert numpy images to tensors and move to GPU
  I, R = convert_to_tensor(I), convert_to_tensor(R)
    
  # Move to GPU if CUDA available
  if torch.cuda.is_available() is True:
    print('Moving images to the GPU...')
    time_to_move_to_GPU = time.time()
    R = R.to(device)
    I = I.to(device)
    
    ## if providing a flat field move that to the GPU also
    try:
        flat = flat[hwidth:nx-hwidth, hwidth:nx-hwidth]
        flat = convert_to_tensor(flat)
        flat = flat.to(device)
    except TypeError:
        pass

  print("--- Time to move data onto GPU: %s ---" % (time.time() - time_to_move_to_GPU))

  kernel, B, model = infer_kernel(R, I, flat, loss_fn,
				   maxiter=max_iterations,
				   FIM=fisher,
                   alpha = alpha,
				   convergence_plots=show_convergence_plots,
				   d = poly_degree,
				   ks = ks,
				   tol = tol,
				   lr_kernel = lr_kernel,
				   lr_B = lr_B,
				   Newton_tol = Newton_tol)


  print("--- Finished in a total of %s seconds ---" % (time.time() - start_time_total))

  return kernel, B, model






