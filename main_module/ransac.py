
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def circle_func(a, b, r, x):
	return (np.sqrt(r**2-(x-a)**2) + b, -np.sqrt(r**2-(x-a)**2) + b)

# Equations for the pointed arch
def pointed_arch (arch_1,arch_2,radius, center_distance, center_x, center_y,threshold):
    """
    This function allow to calculate the parameters of interest for a pointed arch
        
    Parameters:
    ----------
    
    arch_1 (numpy array nx2): array with the x,y coordinates of the points that represent the first quarter of arch
    
    arch_2 (numpy array nx2): array with the x,y coordinates of the points that represent the second quarter of arch
    
    radius (float): radius used for generating the pointed arch
    
    center_distance (float): distance between the center of the first and second quarter of arch
    
    center_x (float): x coodinate of the center of the first quarter of arch
    
    center_y (float): y coordinate of the center of the first quarter of arch
    
    threshold (float): threshold value for consider a point as inlier or outlier
    
    Returns:
    ----------
    
    d (integer): number of inliers according with the RANSAC model
    
    arch_1_inliers (numpy array nx2): array of inliers for the first quarter of arch
    
    arch_1_outliers (numpy array nx2): array of outliers for the first quarter of arch
    
    arch_2_inliers (numpy array nx2): array of inliers for the second quarter of arch
    
    arch_2_outliers (numpy array nx2): array of outliers for the second quarter of arch
    
    x1_masked (list): list of the x coordinates that represnt the best fit first quarter of arch
    
    y1_masked (list): list of the y coordinates that represnt the best fit first quarter of arch
    
    x2_masked (list): list of the x coordinates that represnt the best fit second quarter of arch
    
    y2_masked (list): list of the y coordinates that represnt the best fit second quarter of arch
    
    """
   
    x1_masked,y1_masked,error_1=first_quarter_circle(radius,center_x,center_y,center_distance,True,arch_1[:,0],arch_1[:,1])
    x2_masked,y2_masked,error_2=second_quarter_circle(radius,center_x,center_y,center_distance,True,arch_2[:,0],arch_2[:,1])
    arch_1_inliers=arch_1[np.array(error_1).flatten()<=threshold]
    arch_1_outliers=arch_1[np.array(error_1).flatten()>threshold]
    arch_2_inliers=arch_2[np.array(error_2).flatten()<=threshold]
    arch_2_outliers=arch_2[np.array(error_2).flatten()>threshold]
    d1=len(arch_1_inliers)
    d2=len(arch_2_inliers)
    d=d1+d2
    return d,arch_1_inliers,arch_1_outliers,arch_2_inliers,arch_2_outliers,x1_masked,y1_masked,x2_masked,y2_masked
  
def first_quarter_circle(radius, center_x, center_y, center_distance, compute_error=False, point_x=0, point_y=0):
    """
    This function allow to calculate the curve of the first quarter arch as well as its error
        
    Parameters:
    ----------
      
    radius (float): radius used for generating the pointed arch
    
    center_x (float): x coodinate of the center of the first quarter of arch
    
    center_y (float): y coordinate of the center of the first quarter of arch
    
    center_distance (float): distance between the center of the first and second quarter of arch
    
    compute_error (bool): true if the error is computed. Default: true
    
    point_x (list): list of points (x coordinates) used for calculating the error
    
    point_y (list): list of points (y coordinates) used for calculating the error
    
    Returns:
    ----------
       
    x1_masked (list): list of the x coordinates that represnt the best fit first quarter of arch
    
    y1_masked (list): list of the y coordinates that represnt the best fit first quarter of arch
    
    error (list): list with the value of error of each point (point_x,point_y)
    
    """
    # Plotting the first quarter circle segment
    theta = np.linspace(0, np.pi / 2, 1000)
    x1 = radius * np.cos(theta) + center_x
    y1 = radius * np.sin(theta) + center_y
    # Calculate midpoint between the centers
    mid_x = (center_x + center_x + center_distance) / 2
    mid_y = center_y
    # Calculate the y-coordinate corresponding to the midpoint's x-coordinate on the first circle
    y_mid = np.sqrt(radius ** 2 - (mid_x - center_x) ** 2) + center_y
    # Masking the parts under max_height
    mask1 = y1 <= y_mid
    x1_masked = x1[mask1]
    y1_masked = y1[mask1]    
    error = []
    itera=len(point_x)
    if compute_error:
        pass
        # Calculate the distance between the each point and the curve
        for points in range (itera):
            distances = np.sqrt((x1_masked - point_x[points]) ** 2 + (y1_masked - point_y[points]) ** 2)
            # Find the minimum distance
            min_distance = np.min(distances)
            error.append ([min_distance])  
    
    return x1_masked, y1_masked, error

def second_quarter_circle (radius,center_x,center_y,center_distance,compute_error=False,point_x=0,point_y=0):
    """
    This function allow to calculate the curve of the second quarter arch as well as its error
        
    Parameters:
    ----------
      
    radius (float): radius used for generating the pointed arch
    
    center_x (float): x coodinate of the center of the second quarter of arch
    
    center_y (float): y coordinate of the center of the second quarter of arch
    
    center_distance (float): distance between the center of the first and second quarter of arch
    
    compute_error (bool): true if the error is computed. Default: true
    
    point_x (list): list of points (x coordinates) used for calculating the error
    
    point_y (list): list of points (y coordinates) used for calculating the error
    
    Returns:
    ----------
       
    x2_masked (list): list of the x coordinates that represnt the best fit second quarter of arch
    
    y2_masked (list): list of the y coordinates that represnt the best fit second quarter of arch
    
    error (list): list with the value of error of each point (point_x,point_y)
    
    """
    # Plotting the second quarter circle segment
    theta = np.linspace(0, np.pi/2, 1000)
    x2 = radius * np.cos(np.pi - theta) + center_x + center_distance
    y2 = radius * np.sin(np.pi - theta) + center_y
    # Calculate midpoint between the centers
    mid_x = (center_x + center_x + center_distance) / 2
    mid_y = center_y
    # Calculate the y-coordinate corresponding to the midpoint's x-coordinate on the first circle
    y_mid = np.sqrt(radius**2 - (mid_x - center_x)**2) + center_y
    # Masking the parts under max_height for the second segment
    mask2 = y2 <= y_mid
    x2_masked = x2[mask2]
    y2_masked = y2[mask2]     
    error = []  
    itera=len(point_x)
    if compute_error:
        pass
        # Calculate the distance between the each point and the curve
        for points in range (itera):
            distances = np.sqrt((x2_masked - point_x[points]) ** 2 + (y2_masked - point_y[points]) ** 2)
            # Find the minimum distance
            min_distance = np.min(distances)
            error.append ([min_distance])      
        
    return x2_masked, y2_masked, error

#%% RANSAC FOR CURVE FITTING
class RANSAC:
    """
    Class for RANSAC curve fitting
        
    Parameters:
    ----------
    
    x_data (list): list with the input coordinates of the model [x1
                                                                  x2]
    
    y_data (list): list with the output coordinates of the model [y1
                                                                  y2]
    
    n (int): number of iterations for RANSAC
    
    d_min (int): minimum number of inliers to consider the model as valid. Less number of inliers in a model (iteration) make this model unuseful
    
    dt (float): distance threshold. If a point has a distance to the fitted curve hihger than this value the algorithm will consider it as outlier
    
    type_curve (string): type of curve to fit. Types: Circular arch, Pointed arch
    
    """
    def __init__(self, x_data, y_data, n,d_min,dt,type_curve,midpoint=0):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.d_min = d_min
        self.best_model = None
        self.dt = dt
        self.tc=type_curve
        self.d_best=0
        self.best_x_coordinates = None
        self.best_y_coordinates = None
        if self.tc=="Pointed arch":
            if midpoint==0:
                # Splitting the arch in two parts based on a specific criteria. This is to make a random sample on each half and perform a circle fitting
                midpoint = sum(x_data) / len(x_data) 
            
            # First part of the arch
            filtered_x_data_1 = [x for x in x_data if x >= midpoint]
            filtered_y_data_1 = [y for i, y in enumerate(y_data) if x_data[i] >= midpoint]
            # Combine filtered_x_data and filtered_y_data into a single array (arch_1)
            self.arch_1 = np.array(list(zip(filtered_x_data_1, filtered_y_data_1)))
            
            # Second part of the arch
            filtered_x_data_2 = [x for x in x_data if x < midpoint]
            filtered_y_data_2 = [y for i, y in enumerate(y_data) if x_data[i] < midpoint]
            # Combine filtered_x_data and filtered_y_data into a single array (arch_1)
            self.arch_2 = np.array(list(zip(filtered_x_data_2, filtered_y_data_2)))
            
    def random_sampling(self):
        """
        This function allow to make a random sampling. If we select "circle" in the construction the random sampling will be 3
            
        Returns
        -------
        
        sample (list): list with the random sampled points [x2 y2
                                                                x3 y3]
        """
        sample = []
        save_ran = []
        count = 0
        if self.tc == "Circular arch" or self.tc == "Pointed arch" or self.tc=="Quarter arch":
            max_count=3
            # get three points from data
        while True:
            ran = np.random.randint(len(self.x_data))
            if ran not in save_ran:
                sample.append((self.x_data[ran], self.y_data[ran]))
                save_ran.append(ran)
                count += 1
                if count == max_count:
                    break
        return sample

    def make_model(self, sample):
        """
        This function allow to extract the curve that best fit with respect to the given samples
            
        Parameters
        -------
        
        sample (list): list with the random sampled points [x2 y2
                                                                x3 y3]
        
        Returns
        -------
        
        parameters of the curve. For circle c_x (float),c_y (float),r (float)
        """
        if self.tc == "Circular arch" or self.tc=="Quarter arch":
            # calculate A, B, C value from three points by using matrix
            pt1 = sample[0]
            pt2 = sample[1]
            pt3 = sample[2]
            
            A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]]) 
            B = np.array([[pt2[0]**2 - pt1[0]**2 + pt2[1]**2 - pt1[1]**2], [pt3[0]**2 - pt2[0]**2 + pt3[1]**2 - pt2[1]**2]])		
            inv_A = inv(A)
            
            c_x, c_y = np.dot(inv_A, B) / 2
            c_x, c_y = c_x[0], c_y[0]
            r = np.sqrt((c_x - pt1[0])**2 + (c_y - pt1[1])**2)
            return c_x, c_y, r            
        elif self.tc=="Pointed arch":
            pass
    def eval_model(self, model):
        """
        This function allow to extract the curve that best fit with respect to the given samples
            
        Parameters
        -------
        
        sample (list): list with the random sampled points [x2 y2
                                                                x3 y3]
        
        Returns
        -------
        
        d (int): number of inliers. Higher number indicates better model.
        outliers (list): coordinates of the outliers [x1 y1
                                                      x2 y2]
        inliers (list): coordinates of the inliers [x1 y1
                                                      x2 y2]
        """
        d = 0
        outliers=[]
        inliers=[]
        if self.tc == "Circular arch" or self.tc=="Quarter arch":
            c_x, c_y, r = model
            # Evaluation of the error by measuring the distance of the point with respect to the circle
            for i in range(len(self.x_data)):
                dis = np.sqrt((self.x_data[i]-c_x)**2 + (self.y_data[i]-c_y)**2)
                if dis >= r:
                    distance=dis - r
                else:
                    distance= r - dis           
                if distance > self.dt:
                    out = [self.x_data[i], self.y_data[i]]
                    outliers.append(out)
                else:
                    ins = [self.x_data[i], self.y_data[i]]
                    inliers.append(ins)
        d=len(inliers)
        # Convert inliers and outliers to NumPy arrays for flexibility
        inliers = np.array(inliers)
        outliers = np.array(outliers)
        return d, outliers, inliers

    def execute_ransac(self):
        """
        This function allow to run the RANSAC model over a RANSAC class
        
        Returns
        -------
        
        d_best (int): number of inliers of the best mode
        outliers (list): coordinates of the outliers [x1 y1
                                                      x2 y2]
        inliers (list): coordinates of the inliers [x1 y1
                                                      x2 y2]
        """
        if self.tc == "Circular arch" or self.tc == "Quarter arch":
            # find best model performing n interations
            for i in range(self.n):
                # make one model each new iteration
                model = self.make_model(self.random_sampling())
                d_temp,_,_ = self.eval_model(model)
                # if the number of inliers (d_temp) is largeer than the minimum of inliers that the user define and is larger than the best number of inliers. The model is the better than the previous one
                if self.d_min < d_temp and self.d_best < d_temp:
                    self.best_model = model
                    self.d_best = d_temp
            # Evaluate the best model
            d_best,best_outliers,best_inliers = self.eval_model(self.best_model) 
            
            # get best model from ransac and store the data for plotting the best fit curve
            a, b, r = self.best_model[0], self.best_model[1], self.best_model[2] 
            if self.tc=="Circular arch":
                # Calculate the x and y coordinates of points on the upper half of the circle for plotting
                angles = np.linspace(0, np.pi, 100)
                self.best_x_coordinates = a + r * np.cos(angles)
                self.best_y_coordinates = b + r * np.sin(angles)
            else:
                # Concatenate coordinates to check the direciton of the arch                
                max_y_data=max(self.y_data)
                min_y_data=min(self.y_data)
                max_x_data=self.x_data[self.y_data ==  max_y_data]
                min_x_data=self.x_data[self.y_data ==  min_y_data]
                angles = np.linspace(0,np.pi/2, 100)
                self.best_x_coordinates = a + r * np.cos(angles)
                self.best_y_coordinates = b + r * np.sin(angles)
                if max_x_data>min_x_data:                                  
                    # Horizontal symmetry: Mirror the x-coordinates
                    symmetric_x_coordinates = 2 * a - self.best_x_coordinates
                    self.best_x_coordinates=symmetric_x_coordinates
                              
        elif self.tc=="Pointed arch":
            
            # find best model performing n interations
            for i in range(self.n):

                
                # best fit paramenters for the first quarter of arch based on a random sampling of 3 points
                ransac1 = RANSAC(self.arch_1[:,0],self.arch_1[:,1],1,3,9999,'Circular arch')    
                _,_,_=ransac1.execute_ransac()
                # get the parameters of the circle
                a1, b1, r1 = ransac1.best_model[0], ransac1.best_model[1], ransac1.best_model[2]  

                # best fit paramenters for the second quarter of arch based on a random sampling of 3 points
                ransac2 = RANSAC(self.arch_2[:,0],self.arch_2[:,1],1,3,9999,'Circular arch')
                _,_,_=ransac2.execute_ransac()
                # get the parameters of the circle
                a2, b2, r2 = ransac2.best_model[0], ransac2.best_model[1], ransac2.best_model[2]  

                # average values of both arches to create the pointed arch (best fit arch to the data given)
                rm=(r1+r2)/2
                bm=(b1+b2)/2
                if a1>=a2:
                #If the center of the first quarter is placed before the center of the second quarter is not valid. It is not a pointed arch. We write d=0 (none inliers for RANSAC)
                    d_temp=0
                else:
                # If the first quarter is placed after the second quarter, it could be a pointed arch. So, we perform the curve of the pointed arche as well as its inliers and outliers and number (d) of inliers
                    center_distance=a2-a1
                    d_temp,arch_1_inliers,arch_1_outliers,arch_2_inliers,arch_2_outliers,_,_,_,_=pointed_arch (self.arch_1,self.arch_2,rm, center_distance, a1, bm,self.dt)
                # if the number of inliers (d_temp) is larger than the minimum of inliers that the user define and is larger than the best number of inliers. The model is the better than the previous one
                if self.d_min < d_temp and self.d_best < d_temp:
                    self.best_model = [rm,center_distance,a1,bm]
                    self.d_best = d_temp
            # Evaluate the best model
            d_best,self.arch_1_best_inliers,self.arch_1_best_outliers,self.arch_2_best_inliers,self.arch_2_best_outliers,x_1_masked_best,y_1_masked_best,x_2_masked_best,y_2_masked_best=pointed_arch (self.arch_1,self.arch_2,self.best_model[0], self.best_model[1], self.best_model[2], self.best_model[3],self.dt)
            # Combine the arrays for generating a unique array
            self.best_x_coordinates=np.vstack((x_1_masked_best, x_2_masked_best))
            self.best_y_coordinates=np.vstack((y_1_masked_best, y_2_masked_best))
            best_inliers=np.vstack((self.arch_1_best_inliers, self.arch_2_best_inliers))
            best_outliers=np.vstack((self.arch_1_best_outliers, self.arch_2_best_outliers))                 
   
                    
        return d_best,best_outliers,best_inliers
