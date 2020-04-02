# A set of helpers which are useful for debugging SIFT!
# Feel free to take a look around in case you are curious,
# but you shouldn't need to know exactly what goes on,
# and you certainly don't need to change anything

import numpy as np
import scipy.io as scio

import visualize


# Gives you the TA solution for the interest points you
# should find
def cheat_interest_points(eval_file, scale_factor):
    file_contents = scio.loadmat(eval_file)

    x1 = file_contents['x1']
    y1 = file_contents['y1']
    x2 = file_contents['x2']
    y2 = file_contents['y2']

    x1 = x1 * scale_factor
    y1 = y1 * scale_factor
    x2 = x2 * scale_factor
    y2 = y2 * scale_factor

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)

    return x1, y1, x2, y2


def estimate_fundamental_matrix(Points_a, Points_b):
    # Get linear system of eqns
    # each row will be:
    # [u'u u'v u' v'u v'v v' u v 1]
    # found from rearranging the defn of the fundamental matrix
    # we assume the prime image is image B

    n = Points_b.shape[0]
    u_prime = np.copy(Points_b[:, 0])
    v_prime = np.copy(Points_b[:, 1])
    u = np.copy(Points_a[:, 0])
    v = np.copy(Points_a[:, 1])

    ############################
    # Normalize points
    # Calculate offset matrices combining images a and b
    c_u = np.mean(u)
    c_v = np.mean(v)
    c_u_prime = np.mean(u_prime)
    c_v_prime = np.mean(v_prime)

    offset_matrix = np.array([[1, 0, -c_u], [0, 1, -c_v], [0, 0, 1]])
    offset_matrix_prime = np.array([[1, 0, -c_u_prime], [0, 1, -c_v_prime], [0, 0, 1]])

    # Calculate scale matrices for images a and b
    s = 1 / np.std([[u - c_u], [v - c_v]])
    s_prime = 1 / np.std([[u_prime - c_u_prime], [v_prime - c_v_prime]])

    scale_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    scale_matrix_prime = np.array([[s_prime, 0, 0], [0, s_prime, 0], [0, 0, 1]])

    T_a = scale_matrix @ offset_matrix
    T_b = scale_matrix_prime @ offset_matrix_prime

    # Normalize points from images a and b
    for i in range(0, n):
        norm = T_a @ np.transpose([u[i], v[i], 1])
        norm_prime = T_b @ np.transpose([u_prime[i], v_prime[i], 1])
        u[i] = norm[0]
        v[i] = norm[1]
        u_prime[i] = norm_prime[0]
        v_prime[i] = norm_prime[1]

    # Normalize points ends here
    ############################

    # create data matrix
    data_matrix = np.array([u_prime * u, u_prime * v, u_prime,
                            v_prime * u, v_prime * v, v_prime,
                            u, v, np.ones((n))])
    data_matrix = np.transpose(data_matrix)

    # Get system matrix using svd
    U, S, Vh = np.linalg.svd(data_matrix)

    # Get column of V coresp to the smallest singular value for full rank F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we take the last row instead of the last column
    # Vh is sorted in descending order of the size of the eigenvalues
    # indx = np.argmin(S)
    # full_F = Vh[indx,:]
    full_F = Vh[-1, :]

    # Reshape column to 3x3 so we have the right dimension for F
    full_F = np.reshape(full_F, (3, 3))
    # print(np.linalg.matrix_rank(full_F))

    # Reduce rank to get final F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we don't have to transpose it here.
    # for the matrix multiplication to produce F_matrix
    U, S, Vh = np.linalg.svd(full_F)
    # S is sorted in descending order of the size of the eigenvalues
    # indx = np.argmin(S)
    # S[indx] = 0
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh
    # print(np.linalg.matrix_rank(F_matrix))

    # Adjust back to original coordinates
    F_matrix = np.transpose(T_b) @ F_matrix @ T_a
    return F_matrix


def evaluate_correspondence(img_A, img_B, ground_truth_correspondence_file,
                            scale_factor, x1_est, y1_est, x2_est, y2_est, matches, confidences, vis,
                            filename="notre_dame_matches.jpg"):
    # 'unscale' interest points to compare with ground truth points
    x1_est_scaled = x1_est / scale_factor
    y1_est_scaled = y1_est / scale_factor
    x2_est_scaled = x2_est / scale_factor
    y2_est_scaled = y2_est / scale_factor

    conf_indices = np.argsort(-confidences, kind='mergesort')
    matches = matches[conf_indices, :]
    confidences = confidences[conf_indices]

    # we want to see how good our matches are, extract the coordinates of each matched
    # point

    x1_matches = np.zeros(matches.shape[0])
    y1_matches = np.zeros(matches.shape[0])
    x2_matches = np.zeros(matches.shape[0])
    y2_matches = np.zeros(matches.shape[0])

    for i in range(matches.shape[0]):
        x1_matches[i] = x1_est_scaled[int(matches[i, 0])]
        y1_matches[i] = y1_est_scaled[int(matches[i, 0])]
        x2_matches[i] = x2_est_scaled[int(matches[i, 1])]
        y2_matches[i] = y2_est_scaled[int(matches[i, 1])]

    good_matches = np.zeros((matches.shape[0]), dtype=np.bool)

    # Loads `ground truth' positions x1, y1, x2, y2
    file_contents = scio.loadmat(ground_truth_correspondence_file)

    # x1, y1, x2, y2 = scio.loadmat(eval_file)
    x1 = file_contents['x1']
    y1 = file_contents['y1']
    x2 = file_contents['x2']
    y2 = file_contents['y2']

    pointsA = np.zeros((len(x1), 2))
    pointsB = np.zeros((len(x2), 2))

    for i in range(len(x1)):
        pointsA[i, 0] = x1[i]
        pointsA[i, 1] = y1[i]
        pointsB[i, 0] = x2[i]
        pointsB[i, 1] = y2[i]

    correct_matches = 0

    F = estimate_fundamental_matrix(pointsA, pointsB)
    top50 = 0
    top100 = 0

    for i in range(x1_matches.shape[0]):
        pointA = np.ones((1, 3))
        pointB = np.ones((1, 3))
        pointA[0, 0] = x1_matches[i]
        pointA[0, 1] = y1_matches[i]
        pointB[0, 0] = x2_matches[i]
        pointB[0, 1] = y2_matches[i]

        if abs(pointB @ F @ np.transpose(pointA)) < .1:
            x_dists = x1 - x1_matches[i]
            y_dists = y1 - y1_matches[i]

            # computes distances of each interest point to the ground truth point
            dists = np.sqrt(np.power(x_dists, 2.0) + np.power(y_dists, 2.0))
            closest_ground_truth = np.argmin(dists, axis=0)
            offset_x1 = x1_matches[i] - x1[closest_ground_truth]
            offset_y1 = y1_matches[i] - y1[closest_ground_truth]
            offset_x1 *= img_B.shape[0] / img_A.shape[0]
            offset_y1 *= img_B.shape[0] / img_A.shape[0]
            offset_x2 = x2_matches[i] - x2[closest_ground_truth]
            offset_y2 = y2_matches[i] - y2[closest_ground_truth]
            offset_dist = np.sqrt(np.power(offset_x1 - offset_x2, 2) + np.power(offset_y1 - offset_y2, 2))
            if offset_dist < 70:
                correct_matches += 1
                good_matches[i] = True
        if i == 49:
            print(f'Accuracy on 50 most confident: {int(100 * correct_matches / 50)}%')
            top50 = correct_matches
        if i == 99:
            print(f'Accuracy on 100 most confident: {int(100 * correct_matches / 100)}%')
            top100 = correct_matches

    print(f'Accuracy on all matches: {int(100 * correct_matches / len(matches))}%')

    if vis > 0:
        print("Vizualizing...")
        visualize.show_correspondences(img_A, img_B, x1_est / scale_factor, y1_est / scale_factor,
                                       x2_est / scale_factor, y2_est / scale_factor, matches, good_matches, vis,
                                       filename)

    return top50, top100, correct_matches
