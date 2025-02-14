## In this file I implement the class that defines the manifold
## Specifically, it will include methods for point generations, etc
import numpy as np
import torch
import geometry #Needed for the extrinsic curvature computations

#This is the manifold M3 in Italiano et al.
class M3:
    def __init__(
        self,
        rng = np.random.default_rng(), 
        ## Geometric constants
        L = 1.0, # Length of the basis square
        R = 1.0, #radius of the hemispheres
        eps2 = 0.05,
        k = 0,
        z_cutoff = 0.99, #cutoff in z in percentage of the allowed z range
        dtype = torch.float32,
        sampling_method = "volume" ## can be "volume" to sample in z according to z^{-3}, or "uniform"
        ):

            self.rng = rng
            self.L = L
            self.R = R
            self.eps2 = eps2
            self.k = k ###Denotes the non-diagonal Dehn filling

            self.eps = 1e-10  ## tolerance to check if two vectors are the same

            self.dtype = dtype

            self.n = 3 ##number of dimensions

            #centers of the two hemispheres
            self.center_1 = np.array([0.0, 0.0 , L])
            self.center_2 = np.array([0.0, L ,0.0])

            #x and y lengths of the full reflected polytope
            self.L1 = 4.0*L
            self.L2 = 2.0*L

            self.beta = (4*np.pi)/(self.n-1)
            #rjoin, depends on the length of the filled geodesics
            self.rj = np.sqrt(1 + ((self.k**2)*(self.L1**2)+self.L2**2)/(4*(np.pi**2)))

            if sampling_method == "uniform":
                self.draw_z = self.draw_z_uniform_density
            else:
                self.draw_z = self.draw_z_volume_density

            self.eps_max = z_cutoff*self.rj ##a cut off at rj to avoid silly divergences
            self.refs = [self.ref_F1, self.ref_F2, self.ref_F3] #an ordered list of the reflections needed to build M_3


            ## Now I group all the reflections by color

            self.ref_c1 = [self.ref_F1, self.ref_F12]
            self.ref_c2 = [self.ref_F2]
            self.ref_c3 = [self.ref_F3]

            self.refsAllColors = [self.ref_c1, self.ref_c2, self.ref_c3]

            ## Group by colors the functions that generate the points on the faces

            self.gen_point_c1 = [self.point_on_F1, self.point_on_H2]
            self.gen_point_c2 = [self.point_on_F2, self.point_on_F4]
            self.gen_point_c3 = [self.point_on_F3, self.point_on_H1]

            self.gen_point_colors = [self.gen_point_c1, self.gen_point_c2, self.gen_point_c3]

            # This is a list of normals (which are functions) to each face and of the centers of the boundary hemispheres (None for vertical faces)
            self.list_bry_centers, self.list_normals = self.compute_normals()
            #This is a list of the projectors onto A_i  (eA in the notes)
            self.list_1d_projectors = self.compute_1d_projectors()

            #The tensor that maps projectors on B to projector on A.
            #Using the notation in the appendix, eB = eA@J
            self.Js, self.deltas = self.compute_Js_and_Deltas()

            #This constructs the eB
            self.list_1d_projectors_B = self.compute_mapped_projectors()

    #This generates point distributed accoring to z^{-3}, which is the volume form.
    #Obtained by inverting the CDF (inverse transform sampling)
    def draw_z_volume_density(self, zmin, zmax):
        
        unif_max =  0.5*(zmin**(-2)- zmax**(-2))
        unif = unif_max*self.rng.random()

        return zmin/(np.sqrt(1-2*unif * zmin**2))
    
    def draw_z_uniform_density(self, zmin, zmax):
        return (zmax-zmin)*self.rng.random()+zmin
    
    def point_within_walls(self):
        z_try = self.draw_z(self.eps2, self.eps_max)
        return np.array(
            [z_try,self.L*self.rng.random(),self.L*self.rng.random()])

    #Generates a point above the two hemispheres
    def point_in_P3(self):
        x1 = self.L*self.rng.random()
        x2 = self.L*self.rng.random()

        # Check the z of the hemispheres at this x,y point
        temp = self.R**2-(x1-self.center_1[1])**2-(x2-self.center_1[2])**2
        if temp > 0: z_hem1 = np.sqrt(temp)
        else: z_hem1 = 0.0

        temp = self.R**2-(x1-self.center_2[1])**2-(x2-self.center_2[2])**2
        if temp > 0: z_hem2 = np.sqrt(temp)
        else: z_hem2 = 0.0

        #then check which one is the highest, this will be zmin for drawing the point in z
        zmin = np.max([z_hem1, z_hem2])

        z_try = self.draw_z(zmin, self.eps_max)
    
        return np.array([z_try,x1,x2])

    def above_hem(self, point, center):
        if np.linalg.norm(point-center) >= self.R:
            return True
        else:
            return False

    
    #Now I define the reflections along the faces F1, F2, F3
    def ref_F1(self, point):
        return np.diag([1.0,-1.0,1]).dot(point)
    def ref_F2(self, point):
        return np.diag([1.0,1.0,-1.0]).dot(point)+np.array([0.0, 0.0, 2*self.L])
    def ref_F3(self, point):
        return np.diag([1.0,-1.0,1.0]).dot(point)+np.array([0.0, 2*self.L, 0.0])


    #refs = [ref_F1, ref_F2, ref_F3] #an ordered list of the reflections needed to build M_3


    # Now, starting from a single point, I can generate 8 (= 2^c) points, one on each copy
    def generate_2toc_points(self):
        point = self.point_in_P3()
        points = [point]
        for ref in self.refs: # iterate over the reflections
            new_points = []
            for point in points:
                new_points.append(ref(point))
            points +=  new_points 
        return points

    def draw_points(self):
        return self.generate_2toc_points()

    #generates 8 times npoints
    def generate_bulk_points(self, npoints):
        return np.array([self.draw_points() for _ in range(npoints)]).reshape(-1,3)



    ###############  Point generation for the boundary conditions (gluing)


    #This is the new extra reflection corresponding to face 1
    def ref_F12(self,point):
        return np.diag([1.0,-1.0,1]).dot(point)+np.array([0.0, 4*self.L, 0.0])

    #These are above the hemisphere
    def point_on_F1(self):
        x = 0.0
        y = self.L*self.rng.random()
        zmin = np.sqrt(self.R**2 - (x-self.center_1[1])**2 -(y-self.center_1[2])**2 )
        z_try = self.draw_z(zmin, self.rj)
        return np.array([z_try, x, y])

    def point_on_F2(self):
        x = self.L*self.rng.random()
        y = self.L
        zmin = np.sqrt(self.R**2 - (x-self.center_1[1])**2 -(y-self.center_1[2])**2 )
        z_try = self.draw_z(zmin, self.rj)
        return np.array([z_try, x, y])

    def point_on_F3(self):
        x = self.L
        y = self.L*self.rng.random()
        zmin = np.sqrt(self.R**2 - (x-self.center_2[1])**2 -(y-self.center_2[2])**2 )
        z_try = self.draw_z(zmin, self.rj)
        return np.array([z_try, x, y])

    def point_on_F4(self):
        x = self.L*self.rng.random()
        y = 0.0
        zmin = np.sqrt(self.R**2 - (x-self.center_2[1])**2 -(y-self.center_2[2])**2 )
        z_try = self.draw_z(zmin, self.rj)
        return np.array([z_try, x, y])

    def point_on_H1(self):
        #generate a point in the triangle, and then generate its z
        x1 = self.L*self.rng.random()
        x2 = x1+(self.L-x1)*self.rng.random() ## This is to generate a point between x1 and L
        z = np.sqrt(self.R**2-(x1-self.center_1[1])**2 -(x2-self.center_1[2])**2 )
        return np.array([z,x1,x2])

    def point_on_H2(self):
        #generate a point in the triangle, and then generate its z
        x1 = self.L*self.rng.random()
        x2 = x1*self.rng.random()
        z = np.sqrt(self.R**2-(x1-self.center_2[1])**2 -(x2-self.center_2[2])**2 )
        return np.array([z,x1,x2])

    def generate_2toc_points_from_point(self, point):
        points = [point]
        for ref in self.refs: # iterate over the reflections
            new_points = []
            for point in points:
                new_points.append(ref(point))
            points +=  new_points 
        return points

    #If a point is in the list returns its key. Stops at the first instance.
    def is_point_in_list(self, point, list_to_check):
        key = 0
        for elem in list_to_check:
            if np.linalg.norm(point-elem) < self.eps:
                return key
            key+=1
        return -1
                
        
    #Now I put all the 14 pairs together, and generate n_points_faces of them.
    #Here I write a function that returns two ordered numpy arrays of length 14*n_points_faces,
    #corresponding to the points that have to be identified (point k of array 1 is identified with point k of array 2)
    def generate_boundary_points(self, n_points_faces):
        all_pairs_1 = []
        all_pairs_2 = []

        for i in range(n_points_faces):
            pairs = []
            for color in range(3): #3
                for face_per_color in range(2): #2
                    point = self.gen_point_colors[color][face_per_color]()
                    points_face = self.generate_2toc_points_from_point(point)

                    for vector in points_face:
                        for reflection_fixed_color in self.refsAllColors[color]:
                            temp = reflection_fixed_color(vector) 
                            key = self.is_point_in_list(temp, points_face)
                            if key >= 0:
                                pairs.append([vector, temp])
            #Remove the duplicate pairs and pairs composed of the same point. The result should be 14 pairs
            pairs_dedup = pairs.copy() 
            to_stop = False
            cc = 0
            while (not to_stop):
                cc+=1
                to_stop = True
                for pair in pairs_dedup:
                    key = 0
                    for pair_2 in pairs_dedup:
                        if (np.linalg.norm(pair_2[1] - pair[0]) < self.eps ) and (np.linalg.norm( pair_2[0] - pair[1]) < self.eps):
                            pairs_dedup.pop(key)
                            to_stop = False
                        key+=1
            for pair in pairs_dedup:
                all_pairs_1.append(pair[0])
                all_pairs_2.append(pair[1])
                
        return [np.array(all_pairs_1), np.array(all_pairs_2)]
            
    # Now I reorder them in the following way
    # generate_boundary_points produces the lists of points ordered as (f_1, f2_, ..... f_1, f_2, ..... , f_1, f_2, .....)
    # here I want to order them as ( (f_1, f1_, f_1, ...), (f_2, f_2, f_2, ...), ....)
    def generate_boundary_points_reordered(self, n_points_faces):
        points = self.generate_boundary_points(n_points_faces)
        
        list_fin = [[],[]]

        n_faces = len(points[0])//n_points_faces
        for i in range(n_faces):
            list_1_new = []
            list_2_new = []
            for k in range(n_points_faces):
                list_1_new.append(points[0][i+n_faces*k])
                list_2_new.append(points[1][i+n_faces*k])   
            list_fin[0].append(np.array(list_1_new))
            list_fin[1].append(np.array(list_2_new))
        return list_fin 


####################################### Normals and projectors #################################################################

    
    ## Here I write the code for the projectors.
    ## I define here the e^m_a as functions

    #This one needs to be computed only at initialization of the class
    def compute_1d_projectors(self):

        def proj_fixed_x1(x):
             return torch.tensor(
                [[1.0,0.0,0.0],
                [0.0,0.0,1.0]], device = x.device, dtype = self.dtype)
                
        def proj_fixed_x2(x):
            return torch.tensor(
                    [[1.0,0.0,0.0],
                    [0.0,1.0,0.0]], device = x.device, dtype = self.dtype)
                    
        #projectors on the hemisphere, the first element of the pair
        def proj_sphere_1(x, center):
            c1x1 = center[1]-x[1]
            c2x2 = center[2]-x[2]
            z = x[0]
            return torch.stack([
                    torch.stack([c1x1/z, torch.tensor(1.0, device = x.device), torch.tensor(0.0, device = x.device)]),
                    torch.stack([c2x2/z, torch.tensor(0.0, device = x.device), torch.tensor(1.0, device = x.device)])
            ])



        def create_function_with_fixed_center_1(center):
            def proj_sphere_fixed_1(x):
                return proj_sphere_1(x, center)
            return proj_sphere_fixed_1


        ## Now build the list of projectors from the list of centers of hemispheres


        list_1d_projectors_1 = []
        for center in self.list_bry_centers[0]:
            if center is None:
                list_1d_projectors_1.append(None)
            else:
                list_1d_projectors_1.append(create_function_with_fixed_center_1(center))

        list_1d_projectors_1[4] = proj_fixed_x2
        list_1d_projectors_1[5] = proj_fixed_x2
        list_1d_projectors_1[6] = proj_fixed_x2
        list_1d_projectors_1[7] = proj_fixed_x2
        list_1d_projectors_1[8] = proj_fixed_x1
        list_1d_projectors_1[9] = proj_fixed_x1

        return list_1d_projectors_1    
    
    
    ## I want to add here the code for the normals
    ## Ultimately, I will build a method that returns a list of normals, which are functions
    
    ## Since the normals are fixed, there is not need to execute this method every time
    ## init will execute it and save the list of normal in an attribute

    
    def compute_normals(self):

        ## This are all the possible centers of the hemispheres on which a boundary point can be
        #Some of them are the same because the copies of the pieces of hemisphere can be part of the same big hemisphere (thus same center)

        all_centers = self.generate_2toc_points_from_point(self.center_1)
        all_centers += self.generate_2toc_points_from_point(self.center_2)


        #This function checks if a point belongs to a hemisphere
        def is_point_on_hemi(point, center, radius):
            if (np.linalg.norm(point-center)-radius)**2 < 1e-4:
                return True
            else:
                return False

        #Now I start from a random boundary point with its copies

        generated_points = self.generate_boundary_points(1)

        #now I check which of the boundary points belong to an hemisphere, and record its center

        centers_list_bry_points_1 = []
        for point in generated_points[0]:
            found = False
            for center in all_centers:
                if is_point_on_hemi(point, center, self.R):
                    centers_list_bry_points_1.append(torch.tensor(center, requires_grad= False, dtype = self.dtype, device = 'cpu'))
                    found = True
                    break
            if found == False:
                centers_list_bry_points_1.append(None)

        centers_list_bry_points_2 = []
        for point in generated_points[1]:
            found = False
            for center in all_centers:
                if is_point_on_hemi(point, center, self.R):
                    centers_list_bry_points_2.append(torch.tensor(center, requires_grad= False, dtype = self.dtype, device = 'cpu'))
                    found = True
                    break
            if found == False:
                centers_list_bry_points_2.append(None)


        ## Some functions that create the normals

        def eta_fixed_x1(metric, x):
            g = metric(x)
            eta = torch.zeros(len(x), device = x.device, dtype = self.dtype)
            eta[1] = 1.0

            ## now I normalize it

            #First I need to define eta with index up
            etaU = torch.linalg.solve(g,eta)
            
            #return eta/torch.sqrt(torch.dot(etaU,eta))
            #I am putting an abs because with a generic metric we are not enforcing the signature to be positive
            #Can be simplified if the signature is enforced in the definition of the metric
            return eta/torch.sqrt(torch.abs(torch.dot(etaU,eta)))


        def eta_fixed_x2(metric, x):
            g = metric(x)
            eta = torch.zeros(len(x), device = x.device, dtype = self.dtype)
            eta[2] = 1.0

            ## now I normalize it

            #First I need to define eta with index up
            etaU = torch.linalg.solve(g,eta)
        
            #return eta/torch.sqrt(torch.dot(etaU,eta))
            #I am putting an abs because with a generic metric we are not enforcing the signature to be positive
            #Can be simplified if the signature is enforced in the definition of the metric
            return eta/torch.sqrt(torch.abs(torch.dot(etaU,eta)))


        def eta_sphere(metric, x, center):
            eta = x-center.to(x.device)
            g = metric(x)

            ## now I normalize it

            #First I need to define eta with index up
            etaU = torch.linalg.solve(g,eta)
            
            #return eta/torch.sqrt(torch.dot(etaU,eta))
            #I am putting an abs because with a generic metric we are not enforcing the signature to be positive
            #Can be simplified if the signature is enforced in the definition of the metric
            return eta/torch.sqrt(torch.abs(torch.dot(etaU,eta)))
    

        #This function creates an eta for a sphere with a fixed center
        def create_function_with_fixed_center(center):
            def eta(metric,x):
                return eta_sphere(metric, x, center)
            return eta

        # Now I can create the two lists with the normals (which are functions)
        list_normals_1 = []
        for center in centers_list_bry_points_1:
            if center is None:
                list_normals_1.append(None)
            else:
                list_normals_1.append(create_function_with_fixed_center(center))

        list_normals_2 = []
        for center in centers_list_bry_points_2:
            if center is None:
                list_normals_2.append(None)
            else:
                list_normals_2.append(create_function_with_fixed_center(center))


        list_normals_1[4] = eta_fixed_x2
        list_normals_1[5] = eta_fixed_x2
        list_normals_1[6] = eta_fixed_x2
        list_normals_1[7] = eta_fixed_x2
        list_normals_1[8] = eta_fixed_x1
        list_normals_1[9] = eta_fixed_x1


        list_normals_2[4] = eta_fixed_x2
        list_normals_2[5] = eta_fixed_x2
        list_normals_2[6] = eta_fixed_x2
        list_normals_2[7] = eta_fixed_x2
        list_normals_2[8] = eta_fixed_x1
        list_normals_2[9] = eta_fixed_x1

        centers_list_bry_points_1_numpy = []
        for elem in centers_list_bry_points_1:
            if elem is not None: to_append = elem.numpy()
            else: to_append = None
            centers_list_bry_points_1_numpy.append(to_append)

        centers_list_bry_points_2_numpy = []
        for elem in centers_list_bry_points_2:
            if elem is not None: to_append = elem.numpy()
            else: to_append = None
            centers_list_bry_points_2_numpy.append(to_append)

    
        return [centers_list_bry_points_1_numpy, centers_list_bry_points_2_numpy], [list_normals_1, list_normals_2]

    def compute_Js_and_Deltas(self):
        #compute the affine transfomations that map the points on the first element of the identified pair (p1) to points on the second element of the identified pair (p2)
        #as p2 = p1@J+\Delta 

        Js = [torch.diag(torch.tensor([1,-1,1], dtype=self.dtype)) for i in range(14)]
        Js[4] = Js[5] = Js[6] = Js[7] = Js[8] = Js[9] = torch.diag(torch.tensor([1,1,1], dtype=self.dtype))

        deltas = [(torch.tensor([0,0,2], dtype=self.dtype)) for i in range(14)]
        deltas[0] = deltas[1]= (torch.tensor([0,0,0], dtype=self.dtype))
        deltas[2] = deltas[3] = deltas[8] = (torch.tensor([0,4,0], dtype=self.dtype))
        deltas[9] =  (torch.tensor([0,-4,0], dtype=self.dtype))
        deltas[10] = deltas[11] = deltas[12] =  deltas[13] = (torch.tensor([0,2,0], dtype=self.dtype))

        return Js, deltas

    def compute_mapped_projectors(self):
        #Compute the eB = eA@J
        eBs = []
        for i in range(14):                      #There are 14 pairs of faces
            def mapped_projector(x, i = i):      #without putting the i it alaways references the last value!
                return self.list_1d_projectors[i](x)@((self.Js[i]).to(x.device))
            eBs.append(mapped_projector)
        return eBs






    



