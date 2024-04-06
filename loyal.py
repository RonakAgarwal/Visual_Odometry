import numpy as np

def random_matching_points(matchpts1, matchpts2):
    rand_index = np.random.randint(len(matchpts1), size=8)
    
    X1 = np.array([matchpts1[rand_index[0]], matchpts1[rand_index[1]], matchpts1[rand_index[2]], matchpts1[rand_index[3]], matchpts1[rand_index[4]], matchpts1[rand_index[5]], matchpts1[rand_index[6]], matchpts1[rand_index[7]]]) 
    X2 = np.array([matchpts2[rand_index[0]], matchpts2[rand_index[1]], matchpts2[rand_index[2]], matchpts2[rand_index[3]], matchpts2[rand_index[4]], matchpts2[rand_index[5]], matchpts2[rand_index[6]], matchpts2[rand_index[7]]]) 
    
    return X1, X2

def normalized_f_matrix(points1, points2):
    dist1 = np.sqrt((points1[:, 0] - np.mean(points1[:, 0])) ** 2 + (points1[:, 1] - np.mean(points1[:, 1])) ** 2)
    dist2 = np.sqrt((points2[:, 0] - np.mean(points2[:, 0])) ** 2 + (points2[:, 1] - np.mean(points2[:, 1])) ** 2)

    m_dist1 = np.mean(dist1)
    m_dist2 = np.mean(dist2)
    
    scale1 = np.sqrt(2) / m_dist1
    scale2 = np.sqrt(2) / m_dist2
    
    t1 = np.array([[scale1, 0, -scale1 * np.mean(points1[:, 0])], [0, scale1, -scale1 * np.mean(points1[:, 1])], [0, 0, 1]])
    t2 = np.array([[scale2, 0, -scale2 * np.mean(points2[:, 0])], [0, scale2, -scale2 * np.mean(points2[:, 1])], [0, 0, 1]])

    U_x = (points1[:, 0] - np.mean(points1[:, 0])) * scale1
    U_y = (points1[:, 1] - np.mean(points1[:, 1])) * scale1
    V_x = (points2[:, 0] - np.mean(points2[:, 0])) * scale2
    V_y = (points2[:, 1] - np.mean(points2[:, 1])) * scale2
    
    A = np.zeros((len(U_x), 9))
    
    for i in range(len(U_x)):
        A[i] = np.array([U_x[i] * V_x[i], U_y[i] * V_x[i], V_x[i], U_x[i] * V_y[i], U_y[i] * V_y[i], V_y[i], U_x[i], U_y[i], 1])
       
    U, S, V = np.linalg.svd(A)
    V = V.T
    F = V[:, -1].reshape(3, 3)
    
    Uf, Sf, Vf = np.linalg.svd(F)
    SF = np.diag(Sf)
    SF[2, 2] = 0
    
    F = Uf @ SF @ Vf
    F = t2.T @ F @ t1 
    F = F / F[2, 2]
    
    return F

def compute_fundamental_matrix(U, V):
    max_inliers = 0
    best_inliers_U = []
    best_inliers_V = []
    best_F = None
    
    for i in range(500):
        X1, X2 = random_matching_points(U, V)
        F_r = normalized_f_matrix(X1, X2)
        inliers_U = []
        inliers_V = []
        inliers_count = 0
        
        for j in range(len(U)):
            U1 = np.array([U[j][0], U[j][1], 1]).reshape(1, -1)
            V1 = np.array([V[j][0], V[j][1], 1]).reshape(1, -1)

            epiline1 = F_r @ U1.T
            epiline2 = F_r.T @ V1.T
            error_bottom = epiline1[0] ** 2 + epiline1[1] ** 2 + epiline2[0] ** 2 + epiline2[1] ** 2
            
            error = ((V1 @ F_r @ U1.T) ** 2) / error_bottom
            
            if error[0, 0] < .008:
                inliers_count += 1
                inliers_U.append([U[j][0], U[j][1]])
                inliers_V.append([V[j][0], V[j][1]])
                          
        if max_inliers < inliers_count:
            max_inliers = inliers_count
            best_inliers_U = inliers_U
            best_inliers_V = inliers_V
            best_F = F_r
    
    return best_F




# fundamental_matrix = fundamental(src_pts, dst_pts)