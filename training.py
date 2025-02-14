import os
import sys
import numpy as np
from torch import nn
from torch.func import vmap, vjp, jacrev

#from inflation import *
import geometry 
import manifolds
import metrics
from dataclasses import dataclass
import NNs
from itertools import chain
import time
from parser import create_parser
from inflation import *
from utils import *

import shutil

dtype = torch.float32
torch.set_float32_matmul_precision('high') #uses tensorFloat32 for the intermediate computations if the GPU supports it



if not os.path.exists('./saved-models'):
    os.makedirs('./saved-models')


parser = create_parser()
args = parser.parse_args()


if args.debug == "true":
    use_wandb = False
    debug = True
else:
    use_wandb = True
    debug = False

if args.use_wandb == "true": use_wandb = True
else: use_wandb = False

if args.loadname == None: pre_train = True
else: pre_train = False


if args.load_opt == "true": load_opt_state = True
else: load_opt_state = False

args_dict = vars(args).copy()

if args.cpu == "true":
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#----- Define the NNs we will use for each metric component -----#

@dataclass
class network_config:
    input_dim:        int = 3
    output_dim:       int = 1
    width:            int = args.width       
    depth:            int = args.depth 
    activation:       str = args.activation #can be gelu or  tanh    

#Define the models
models_to_pre_train = [NNs.Res_FC(network_config).to(device) for _ in range(3)]    #These are the ones not initialized to zero
models_to_pre_train.append(NNs.Res_FC_zero(network_config).to(device))
models_zero = [NNs.Res_FC_zero(network_config).to(device) for _ in range(2)]   #These are the ones initialized to zero

models = models_to_pre_train+models_zero




print(f"Total number of parameters: {sum(parameter.numel() for model in models for  parameter in model.parameters() if parameter.requires_grad)}")

#----- Initialize the manifold -----#

k = args.k                                  #controls length of the geodesic used for the filling
z_cutoff = args.cutoff                      #cutoff in 0, as a percentage of z_max
sampling_method = args.sampling             #sampling strategy in z, can be "volume" or "uniform"

m3 = manifolds.M3(sampling_method = sampling_method, k = k, z_cutoff = z_cutoff, dtype = dtype)

if args.pretraining_uniform == 'true' and sampling_method == 'volume':
    m3_pre = manifolds.M3(sampling_method = 'uniform', k = k, z_cutoff = z_cutoff, dtype = dtype)
else:
    m3_pre = m3

#----- Initialize the metrics -----#

pre_training_target = metrics.approximate_DF(m3, dtype = dtype)     #The piece-wise starting point
pre_training_target_batched = vmap(pre_training_target.metric_DF)   

metric_from_NN = metrics.non_diagonal_metric(m3, models, dtype = dtype)  #The metric we are solving for
metric_from_NN_batched = vmap(metric_from_NN.metric)    


if use_wandb:
    import wandb    
    project_name = args_dict.pop("project")

#----- Pre Training ------#

if pre_train:
    print("Pre-training:")

    if use_wandb:
        # --- Initialize wandb --- #
        run = wandb.init(
            project = project_name+"-pretraining",
            config = args_dict
        )
        experiment_name = str(run.id)
    else:
        experiment_name = args.name

    name_prefix = "saved-models/"+experiment_name

    batch_size_pre = args.batch_size_pretraining #remember: the actual number of points is 8 times this number
    n_batches_pre = int(1e6)
    threshold_stop_pre = args.threshold_pretraining
    jacobian_weight = 1
    hessian_weight = 1
    if debug == True:
        fraction_points_jacobian = 0.01#  #fraction of the points on which to also enforce matching of jacobians
        fraction_points_hessian = 0.01#   #fraction of the points on which to also enforce matching of hessians (it's very expensive to compute it for all points)
    else:
        fraction_points_jacobian = 0.5  #fraction of the points on which to also enforce matching of jacobians
        fraction_points_hessian = 0.3   #fraction of the points on which to also enforce matching of hessians (it's very expensive to compute it for all points)
        
    lr_pre = args.lr_pre
    criterion = nn.SmoothL1Loss()

    optimizer_pre = torch.optim.Adam(chain(*[model.parameters() for model in models_to_pre_train]), lr = lr_pre)


    def compute_jacobians_and_hessians(func, inputs):
        jacobian_func = vmap(jacrev(func))
        func_batched = vmap(func)
        outputs = func_batched(inputs)
        jacobians = jacobian_func(inputs[:int(fraction_points_jacobian*len(inputs))])
        hessian_func = vmap(jacrev(jacrev(func)))
        hessians = hessian_func(inputs[:int(fraction_points_hessian*len(inputs))])
        return outputs, jacobians, hessians

    stop = False
    losses_fun, losses_jac, losses_hess, losses_all = [], [], [], [] 
    iterations = []
    best_loss = 1e20
    for epoch in range(n_batches_pre):
        if stop == True:
            break
        inputs = torch.tensor(m3_pre.generate_bulk_points(batch_size_pre), dtype=dtype, device=device, requires_grad = True)
        
        target_outputs, target_jacobians, target_hessians = compute_jacobians_and_hessians(pre_training_target.metric_DF, inputs)
        
        optimizer_pre.zero_grad()
        model_outputs, model_jacobians, model_hessians = compute_jacobians_and_hessians(metric_from_NN.metric, inputs)
        # Compute losses
        output_loss = criterion(model_outputs, args.conformal_factor*target_outputs)/args.conformal_factor
        jacobian_loss = criterion(model_jacobians, args.conformal_factor*target_jacobians)/args.conformal_factor
        hessian_loss = criterion(model_hessians, args.conformal_factor*target_hessians)/args.conformal_factor
        
        loss = (output_loss + jacobian_weight * jacobian_loss+ hessian_weight * hessian_loss)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN, stopping execution.")
            sys.exit(0)
    
        loss.backward()
        def closure():
            return loss
        optimizer_pre.step(closure)

        if device == torch.device('cuda'): torch.cuda.synchronize()     
            
        if epoch%10 ==0:
            with torch.no_grad():

                if use_wandb:
                    wandb.log({"train_loss_pretraining": loss.item()})
                    wandb.log({"output_loss_pretraining": output_loss.item()})
                    wandb.log({"jacobian_loss_pretraining": jacobian_loss.item()})
                    wandb.log({"hessian_loss_pretraining": hessian_loss.item()})
                    
                losses_fun.append(output_loss.item())
                losses_jac.append(jacobian_loss.item())
                losses_hess.append(hessian_loss.item())
                losses_all.append(loss.item())
                iterations.append(epoch)
                
                print("Train loss:\t", loss.item())
                print("Output loss:\t", output_loss.item())
                print("Jacobian loss:\t", jacobian_loss.item())
                print("Hessian loss:\t", hessian_loss.item())
                
                if loss.item()< threshold_stop_pre:
                    stop = True
                    break

                #Save the best
                if loss.item() < best_loss:
                    best_loss = loss.item()

                    for i in range(len(models)): torch.save(models[i].state_dict(), f'{name_prefix}-model{i}.pth')
                    #save the corresponding optimizer states
                    torch.save(optimizer_pre.state_dict(), f'{name_prefix}-optimizer.pth')
                    

    optimizer_pre.zero_grad()

    if use_wandb: run.finish()

    #create a copy of the pre-trained model for re-use
    for i in range(len(models)): shutil.copy(name_prefix+f'-model{i}.pth', f'{name_prefix}-pretrained-model{i}.pth')
    name_pretrained = f'{name_prefix}-pretrained'
torch.cuda.empty_cache()

#----- Training -----#
print("Training:")

if use_wandb:
    # --- Initialize wandb --- #
    run = wandb.init(
        project = project_name,
        config = args_dict
    )
    experiment_name = str(run.id)
else:
    experiment_name = args.name

name_prefix = "saved-models/"+experiment_name

# Loading the networks
if pre_train: loadname = name_pretrained
else: loadname = 'saved-models/'+args.loadname

for i in range(len(models)): models[i].load_state_dict(torch.load(f'{loadname}-model{i}.pth', map_location = device.type, weights_only=True))

if args.norm_type == "L1":
    criterion = MeanRelativeError(norm_type = "L1")
elif args.norm_type == "L2":
    criterion = MeanRelativeError(norm_type = "L2")        


batch_size_bulk = args.batch_size_bulk
batch_size_bry = args.batch_size_bry


Ricci_weight = args.Rweight
h_weight = args.hweight
Kab_weight = args.Kweight
n_batches = int(1e6)

lr = args.lr 
wd = args.wd
all_parameters_iter = chain(*[model.parameters() for model in models])
all_parameters = [para for para in all_parameters_iter]


#Initialize optimizer
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(all_parameters, lr = lr, weight_decay = wd, betas = (args.beta1,args.beta2))
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(all_parameters, lr = lr, weight_decay = wd, momentum = args.momentum)
elif args.optimizer == "ECDSep_scaled":
    optimizer = ECDSep_scaled(all_parameters, lr = lr, weight_decay = wd, nu = args.nu, eta = args.eta, F0 = args.F0)

#load the optimizer state
if load_opt_state: optimizer.load_state_dict(torch.load(f'{loadname}-optimizer.pth'))


#Batched version of functions needed
compute_christoffel_batched = vmap(metric_from_NN.compute_christoffel)
compute_diff_christoffel_batched = vmap(metric_from_NN.der_christoffel)

ricci_batched = vmap(geometry.ricci)
ricci_scalar_batched = vmap(geometry.ricci_scalar)

#here I batch the 1d projectors on the boundaries
list_1d_projectors_batched = []
for elem in m3.list_1d_projectors:
    list_1d_projectors_batched.append(vmap(elem))

list_1d_projectors_batched_B = []
for elem in m3.list_1d_projectors_B:
    list_1d_projectors_batched_B.append(vmap(elem))

projector_fun_batched = vmap(geometry.project_2_tensor)

normals_bry_A, normals_bry_B  = m3.list_normals



##For the boundary, half of the importance points will come from h, the other half from K
##I am also doing it face by face. That is, I am keeping the points with maximum error for each face, not among all the faces
resampling = False
resampling_bry = False
change_fixed_points_every = args.resampling_frequency   #how often change the points used for importance sampling
    
if args.resampling == "importance" or args.resampling_bry == "importance":
    fraction_original_distribution_points = 1 - args.resampling_fraction    #fraction of points kept with the original distribution 
    
    if args.resampling == "importance":
        batch_size_bulk_total = batch_size_bulk
        batch_size_bulk = int(fraction_original_distribution_points*batch_size_bulk_total)
        batch_size_importance_points = batch_size_bulk_total - batch_size_bulk

        #These are initial ones
        bulk_points_importance = m3.generate_bulk_points(batch_size_importance_points)        #(8*batch_size_importance_points, 3) numpy array
        resampling = True
    if args.resampling_bry == "importance":
        batch_size_bry_total = batch_size_bry
        batch_size_bry = int(fraction_original_distribution_points*batch_size_bry_total)
        batch_size_importance_points_bry = batch_size_bry_total - batch_size_bry
        #These are initial ones
        bry_points_importance = m3.generate_boundary_points_reordered(batch_size_importance_points_bry)  #[2,14] list of (batch_size_bry, 3) numpy arrays
        resampling_bry = True


iterations = []
losses_Ricci, losses_h, losses_Kab, losses = [], [], [], []
loss_Ricci = torch.tensor(0., device = device, dtype = dtype)

best_loss = 1e20
if debug:
    bulk_points = m3.generate_bulk_points(batch_size_bulk) 
    bulk_points = torch.tensor(bulk_points, device = device, dtype = dtype, requires_grad = True)
for epoch in range(n_batches):
    t_init = time.time()
    optimizer.zero_grad()
    if not debug:
        bulk_points = m3.generate_bulk_points(batch_size_bulk)              #(8*batch_size_bulk, 3) numpy array
    
        if resampling: bulk_points = np.vstack((bulk_points, bulk_points_importance))

        bulk_points = torch.tensor(bulk_points, device = device, dtype = dtype, requires_grad = True)

    bry_points = m3.generate_boundary_points_reordered(batch_size_bry)  #[2,14] list of (batch_size_bry, 3) numpy arrays
       
    if resampling_bry: bry_points = [ [np.vstack((bry_points[iside][ifa], bry_points_importance[iside][ifa])) for ifa in range(14)] for iside in range(2)]
    
   
    #Check if we have to resample the importance points in this epoch
    if  (epoch%change_fixed_points_every == change_fixed_points_every-1):
        
        if resampling_bry:
            to_resample_bry = True
            new_importance_points_face_h_A = []
            new_importance_points_face_h_B = []
            new_importance_points_face_K_A = []
            new_importance_points_face_K_B = []
        else: to_resample_bry = False

        if resampling: to_resample = True
        else: to_resample = False

    else:
        to_resample_bry = False
        to_resample = False


    # --- Ricci loss --- #
    t0 = time.time()
    gmn = metric_from_NN_batched(bulk_points)        
    christ = compute_christoffel_batched(bulk_points)
    diff_christ = compute_diff_christoffel_batched(bulk_points)

    Rmn = ricci_batched(christ, diff_christ)
    loss_Ricci = criterion(Rmn, -2*(1/args.conformal_factor)*gmn)
    if to_resample:
        print("Resampling importance points")
        with torch.no_grad():
            #new_selected_bulk_points = criterion.return_ordered_points(Rmn, -2*gmn, bulk_points)
            new_selected_bulk_points = criterion.return_ordered_points(Rmn, -2*(1/args.conformal_factor)*gmn, bulk_points)
            bulk_points_importance = new_selected_bulk_points[:8*batch_size_importance_points].cpu().numpy()
    print(f"Time for computing the Ricci tensor: {(time.time()-t0):2f}")
    # --- Boundary loss --- #
    loss_h = torch.tensor(0, device = device, dtype = dtype)
    loss_Kab = torch.tensor(0, device = device, dtype = dtype)
    t0 = time.time()
    
    #put all the bry points on each side in a numpy array
    #For each of them, the first index is ifa, the face index

    all_bry_points_np_0 = np.stack(bry_points[0])
    all_bry_points_np_1 = np.stack(bry_points[1])
    points_A_all = torch.tensor(all_bry_points_np_0,  dtype = dtype, device = device, requires_grad = True)
    points_B_all = torch.tensor(all_bry_points_np_1,  dtype = dtype, device = device, requires_grad = True)
    

    #Now evalute the metric in batches and then split the face index again
    #Also put the face index as first
    gmnA_all =  metric_from_NN_batched(points_A_all.view(-1,3)).view(14,-1,3,3)
    gmnB_all =  metric_from_NN_batched(points_B_all.view(-1,3)).view(14,-1,3,3)
    
    christ_face_A_all = compute_christoffel_batched(points_A_all.view(-1,3)).view(14,-1,3,3,3)
    christ_face_B_all = compute_christoffel_batched(points_B_all.view(-1,3)).view(14,-1,3,3,3)
        

    for ifa in range(14):
        
        points_A = points_A_all[ifa]
        points_B = points_B_all[ifa]
        gmnA = gmnA_all[ifa]
        gmnB = gmnB_all[ifa]
        christ_face_A = christ_face_A_all[ifa]
        christ_face_B = christ_face_B_all[ifa]


        #projectors on the two faces
        projectors_1d_evaluated_A = list_1d_projectors_batched[ifa](points_A)
        #Notice that also the eB need to be evaluated at the same points_A according to our formulas
        projectors_1d_evaluated_B = list_1d_projectors_batched_B[ifa](points_A)

        #-- continuity --#
        hab_A = projector_fun_batched(projectors_1d_evaluated_A,gmnA)
        hab_B = projector_fun_batched(projectors_1d_evaluated_B,gmnB)
        loss_h += criterion(hab_A, hab_B)/14.
        #loss_h += torch.linalg.norm(hab_A-hab_B)/(hab_A.numel()*14)

        if to_resample_bry:
            print("Resampling importance points bry, continuity")
            with torch.no_grad():
                temp_points_A, temp_points_B = criterion.return_ordered_points_bry(hab_A, hab_B, points_A, points_B)
                new_importance_points_face_h_A.append(temp_points_A[:batch_size_importance_points_bry//2].cpu().numpy())
                new_importance_points_face_h_B.append(temp_points_B[:batch_size_importance_points_bry//2].cpu().numpy())
                # new_selected_bulk_points = criterion.return_ordered_points(Rmn, -2*gmn, bulk_points)
                # bulk_points_importance = new_selected_bulk_points[:8*batch_size_importance_points].cpu().numpy()

        #-- Extrinsic curvature --#
        #Fixing the metric in the normal 1 form
        def norm_A_fixed_metric(x):
            return normals_bry_A[ifa](metric_from_NN.metric, x)
        def norm_B_fixed_metric(x):
            return normals_bry_B[ifa](metric_from_NN.metric, x)

        #fixing the function (because cannot batch along functions)
        def covD_down_face_A(christ, x):
            return geometry.covD_down(norm_A_fixed_metric, christ, x)

        def covD_down_face_B(christ, x):
            return geometry.covD_down(norm_B_fixed_metric, christ, x)

        #batching
        covD_down_face_A_batched  = vmap(covD_down_face_A)
        covD_down_face_B_batched  = vmap(covD_down_face_B)
        tc = time.time()
        #compute the extrinsic curvatures FIXME: This is missing a,u, but it seems it doesn't matter when projected
        Kmn_A = covD_down_face_A_batched(christ_face_A, points_A)
        Kmn_B = covD_down_face_B_batched(christ_face_B, points_B)

        #Now project onto the hypersurface
        Kab_A = projector_fun_batched(projectors_1d_evaluated_A,Kmn_A)
        Kab_B = projector_fun_batched(projectors_1d_evaluated_B,Kmn_B)
        loss_Kab += criterion(Kab_A, Kab_B)/14.
            

        if to_resample_bry:
            print("Resampling importance points bry, extrinsic curvature")
            with torch.no_grad():
                temp_points_A, temp_points_B = criterion.return_ordered_points_bry(Kmn_A, Kmn_B, points_A, points_B)
                new_importance_points_face_K_A.append(temp_points_A[:batch_size_importance_points_bry//2].cpu().numpy())
                new_importance_points_face_K_B.append(temp_points_B[:batch_size_importance_points_bry//2].cpu().numpy())
          
    if to_resample_bry:
        #for each face, stack together the selected points with high h and with high K
        bry_points_importance = [
            [np.vstack((new_importance_points_face_h_A[ifa], new_importance_points_face_K_A[ifa])) for ifa in range(14)],
            [np.vstack((new_importance_points_face_h_B[ifa], new_importance_points_face_K_B[ifa])) for ifa in range(14)]
        ]
    print(f"Time for computing the boundary losses: {(time.time()-t0):2f}")
    loss = Ricci_weight*loss_Ricci+h_weight*loss_h+Kab_weight*loss_Kab
    t0 = time.time()
    loss.backward()
    def closure():
        return loss
    print(f"Time for computing the backprop: {(time.time()-t0):2f}")
    t0 = time.time()
    optimizer.step(closure)
    print(f"Time for the optimizer step: {(time.time()-t0):2f}")
    
    # --- Printing and saving --- #
    if epoch%1 == 0:
        with torch.no_grad():
            iterations.append(epoch)
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
                wandb.log({"ricci_loss": loss_Ricci.item()})
                wandb.log({"h_loss": loss_h.item()})
                wandb.log({"K_loss": loss_Kab.item()})
                

            losses_Ricci.append(loss_Ricci.item())
            losses_h.append(loss_h.item())
            losses_Kab.append(loss_Kab.item())
            losses.append(loss.item())
            print(f"Loss Total:\t\t {loss.item():.4e}")
            print(f"Loss Ricci:\t\t {loss_Ricci.item():.4e}")
            print(f"Loss Kab:\t\t {loss_Kab.item():.4e}")
            print(f"Loss h:\t\t\t {loss_h.item():.4e}")
            print(f"Time per iteration: {time.time()-t_init}")
            

        #Save the best
        if loss.item() < best_loss:
            best_loss = loss.item()
            for i in range(len(models)): torch.save(models[i].state_dict(), f'{name_prefix}-model{i}.pth')
            #Save also the optimizer state
            torch.save(optimizer.state_dict(), f'{name_prefix}-optimizer.pth')
