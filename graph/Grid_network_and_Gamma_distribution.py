import numpy as np

# Adjacency Matrix
class Matrix():
    def __init__(self, line_numbers, column_numbers, link_mean_min = 5, link_mean_max=50, link_var_min = 1, link_var_max = 10):
        self.adj_matrix = None  # adj matrix
        self.link_mean = None   # matrix with means
        self.link_var = None    # matrix with variances
        self.line_numbers = line_numbers
        self.column_numbers = column_numbers
        self.link_mean_min = link_mean_min
        self.link_mean_max = link_mean_max
        self.link_var_min = link_var_min
        self.link_var_max = link_var_max
    
    def get_adjmatrix(self):
        return self.adj_matrix
    
    def get_linkmean(self):
        return self.link_mean

    def get_linkvar(self):
        return self.link_var

    def build_adj(self):
        # build matrix
        self.adj_matrix = np.zeros((self.line_numbers * self.column_numbers,self.line_numbers * self.column_numbers))
        #
    ############### line 0
    #
        self.adj_matrix[0,1]=1
        self.adj_matrix[0,self.column_numbers] = 1
        for j in range(1,self.line_numbers-1):
            self.adj_matrix[j,j-1] = 1
            self.adj_matrix[j,j+1] = 1
            self.adj_matrix[j,j+self.column_numbers] = 1				
        self.adj_matrix[self.column_numbers-1,self.column_numbers-2] = 1
        self.adj_matrix[self.column_numbers-1,2*self.column_numbers-1] = 1
        #
        ############## lines 1 to n-2
        #
        for i in range(1,self.line_numbers-1):
            self.adj_matrix[i*self.column_numbers,i*self.column_numbers+1] = 1
            self.adj_matrix[i*self.column_numbers,(i-1)*self.column_numbers] = 1
            self.adj_matrix[i*self.column_numbers,(i+1)*self.column_numbers] = 1
            for j in range(1,self.line_numbers-1):
                self.adj_matrix[i*self.column_numbers+j,i*self.column_numbers+j-1] = 1
                self.adj_matrix[i*self.column_numbers+j,(i-1)*self.column_numbers+j] = 1
                self.adj_matrix[i*self.column_numbers+j,i*self.column_numbers+j+1] = 1			
                self.adj_matrix[i*self.column_numbers+j,(i+1)*self.column_numbers+j] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,(i+1)*self.column_numbers-2] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,i*self.column_numbers-1] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,(i+2)*self.column_numbers-1] = 1
        #
        ################ line n-1
        #
        self.adj_matrix[(self.line_numbers-1)*self.column_numbers,(self.line_numbers-2)*self.column_numbers] = 1
        self.adj_matrix[(self.line_numbers-1)*self.column_numbers,(self.line_numbers-1)*self.column_numbers+1] = 1
        for j in range(1,self.line_numbers-1):
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-1)*self.column_numbers+j-1] = 1
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-2)*self.column_numbers+j] = 1		
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-1)*self.column_numbers+j+1] = 1
        self.adj_matrix[self.line_numbers*self.column_numbers-1,self.line_numbers*self.column_numbers-2] = 1
        self.adj_matrix[self.line_numbers*self.column_numbers-1,(self.line_numbers-1)*self.column_numbers-1] = 1


    # Build link mean travel time matrix

    def build_link_mean(self):
        self.link_mean = self.link_mean_min + (self.link_mean_max - self.link_mean_min) * self.adj_matrix*np.random.random((self.line_numbers*self.column_numbers,self.line_numbers*self.column_numbers))

    # Build link variance travel time matrix

    def build_link_var(self):
        self.link_var = self.link_var_min +  (self.link_var_max - self.link_var_min) * self.adj_matrix*np.random.random((self.line_numbers*self.column_numbers,self.line_numbers*self.column_numbers))



# generate Travel Times following Gamma distribution: 
#        shape = pow(self.link_mean[i][j],2)/self.link_var[i][j]
#        scale = self.link_var[i][j]/self.link_mean[i][j]
#        self.travel_time = np.random.gamma(shape, scale, 1)[0]
  
    def compute_mean_matrix(self):
        return self.adj_matrix * self.link_mean
    
    def compute_variance_matrix(self):
        return self.adj_matrix * self.link_var

    def compute_matrices(self):
        """
        Returns both the mean and variance matrices.
        """
        self.build_adj()
        self.build_link_mean()
        self.build_link_var()
        return self.compute_mean_matrix(), self.compute_variance_matrix()
    
