import numpy as np
import scipy.linalg as sp
import cv2 as cv
import random
import sys


def find_sift_points_for_first_frame(query, x, y, w, h):
    sift = cv.xfeatures2d.SIFT_create()
    mask = np.zeros(query.shape[:2], dtype=np.uint8)
    cv.rectangle(mask, (x,y), (x+w, y+h), (255), thickness = -1)
    keypoints_1, descriptors_1 = sift.detectAndCompute(query, mask)
    return keypoints_1, descriptors_1

def find_matches(keypoints_1, descriptors_1, img, ratio):
    sift = cv.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = sift.detectAndCompute(img, None)

    # matching with knn from
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Filter matches with ratio, store good matches
    good_matches = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return keypoints_1, keypoints_2, good_matches


# Self implemented RANSAC and predict affine tranformation
# Function to computer max interation number
def max_iterations(P, p, s):
    return np.ceil(np.log(1 - P) / np.log(1 - np.power(p, s)))


def two_norm(x):
    return np.sqrt(x.dot(x))


def check_inlier(point_1, point_2, affine, t_fit):
    # if point fits model with an error smaller than t add 1 to inliers
    approx_dst = np.matmul(affine, np.append(point_1, 1.))
    error = two_norm(approx_dst - point_2)
    return error < t_fit


def ransac_affine(src_pts, dst_pts, min_iter):
    # probability for inliers in the data points
    p_inliers = 0.6
    # probability for finding the right answer
    P_right = 0.98
    # sample number of each trial
    sample_size = 3
    # threshold for determining whether the data point fit the computed affine transformation
    t_fit = 5
    # max number of iteration
    max_iter = max_iterations(P_right, p_inliers, sample_size)
    # print("max_inter: {}".format(max_iter))
    # the number of close data values required to assert that a model fits well to the data
    t_inliers = int(len(src_pts) * p_inliers)

    # init model with consideration of all matches
    best_model = np.empty((2, 3))
    most_inliers = 0
    iteration = 0

    while iteration < max_iter or iteration < min_iter:
        # print(iteration)
        # s randomly selected values from data
        indexs = [random.randint(0, len(src_pts) - 1) for s in range(sample_size)]
        inliers = sample_size
        # print(indexs)

        # compute affine transformation with n sampled data
        sample_src = np.array([src_pts[i] for i in indexs]).astype(np.float32)
        sample_dst = np.array([dst_pts[i] for i in indexs]).astype(np.float32)
        model = cv.getAffineTransform(sample_src, sample_dst)

        # check inliers by:
        # for every point in data
        inlier_matrix = np.array([check_inlier(src_pts[i], dst_pts[i], model, t_fit) for i in range(len(src_pts))])
        inliers = np.sum(inlier_matrix)

        # only keep the affine if it has more inliers or have smaller error
        if inliers > most_inliers:
            best_model = model
            most_inliers = inliers

        # stop the loop if we have a model that is good enough
        if most_inliers > len(src_pts) * p_inliers:
            break

        iteration += 1

    return max_iter, t_inliers, best_model, most_inliers


def find_corner(affine, corners):
    new_corners = []
    for corner in corners:
        estimate = np.matmul(affine, corner)
        new_corners.append((int(estimate[0]), int(estimate[1])))
    return new_corners


if __name__ == '__main__':
    total_args = len(sys.argv)
    video_path = "../starbucks/jimin_2.mp4"
    output_path = "../result/jimin_output.m4v"
    svm_path = ""

    if total_args > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        # svm_folder = sys.argv[3]

    min_matches_required = 10

    # create a VideoCapture object
    cap = cv.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv.VideoCapture(video_path)
        cv.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    print(pos_frame)
    # print out total frame number
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print(total_frames)
    # get frame per second
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    # get frame height and width
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    # cap.set(3, 640)
    # cap.set(4, 480)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    output = cv.VideoWriter(output_path, fourcc, 20.0, (width, height))

    x = 0
    y = 0
    w = 0
    h = 0

    keypoints_1 = None
    descriptors_1 = None
    # read first frame
    flag, first_frame = cap.read()
    corners = []
    if flag:
        print("first frame")
        # cat folder
        # x = 460
        # y = 80
        # w = 340
        # h = 470

        # jimin
        x = 560
        y = 100
        w = 150
        h = 425
        corners = np.array([[x, y, 1.], [x + w, y, 1.], [x, y + h, 1.], [x + w, y + h, 1.]])
        # cv.rectangle(first_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
        cv.imwrite("../starbucks/first_frame.jpg", first_frame)
        keypoints_1, descriptors_1 = find_sift_points_for_first_frame(first_frame, x, y, w, h)
        # write the first frame
        output.write(first_frame)
    else:
        print("Error: could not read first frame.")

    print(cap.get(cv.CAP_PROP_POS_FRAMES))

    # flag, sec_frame = cap.read()
    # if flag:
    #     print("sec frame")
    #     # x = 460
    #     # y = 80
    #     # w = 340
    #     # h = 470
    #     # cv.rectangle(first_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
    #     # cv.imwrite("../starbucks/first_frame.jpg", first_frame)
    #     # keypoints_1, descriptors_1 = find_sift_points_for_first_frame(first_frame, x, y, w, h)
    #     # # write the first frame
    #     # output.write(first_frame)
    #     keypoints_1, keypoints_2, good_matches = find_matches(keypoints_1, descriptors_1, sec_frame, 0.60)
    #     print(len(good_matches))
    #     matched_img = np.empty((max(first_frame.shape[0], sec_frame.shape[0]), first_frame.shape[1] + sec_frame.shape[1], 3),
    #                            dtype=np.uint8)
    #     cv.drawMatches(first_frame, keypoints_1, sec_frame, keypoints_2, good_matches, matched_img,
    #                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #
    #     cv.imwrite("../starbucks/matches.jpg", matched_img)
    #
    # else:
    #     print("Error: could not read first frame.")


    # loop over following frames for each frame compute affine transformation
    # and draw rectangle
    # while True:
    #     flag, frame = cap.read()
    #     if flag:
    #         print(cap.get(cv.CAP_PROP_POS_FRAMES))
    #
    #         keypoints_1, keypoints_2, good_matches = find_matches(keypoints_1, descriptors_1, frame, 0.50)
    #         if len(good_matches) < min_matches_required:
    #             print("Error: No enough matches")
    #             continue
    #
    #         src_pts = np.array([np.round(keypoints_1[m.queryIdx].pt) for m in good_matches])
    #         dst_pts = np.array([np.round(keypoints_2[m.trainIdx].pt) for m in good_matches])
    #
    #         max_iter, t_inliers, model, inliers = ransac_affine(src_pts, dst_pts, 1000)
    #
    #         # new_frame = frame.copy()
    #         new_corners = find_corner(model, corners)
    #
    #         cv.line(frame, tuple(new_corners[0]), tuple(new_corners[1]), (255, 255, 0), 3)
    #         cv.line(frame, tuple(new_corners[0]), tuple(new_corners[2]), (255, 255, 0), 3)
    #         cv.line(frame, tuple(new_corners[1]), tuple(new_corners[3]), (255, 255, 0), 3)
    #         cv.line(frame, tuple(new_corners[2]), tuple(new_corners[3]), (255, 255, 0), 3)
    #
    #         # The frame is ready and already captured
    #         key = cv.waitKey(33) & 0xFF
    #         output.write(frame)
    #
    #     else:
    #         # The next frame is not ready, so we try to read it again
    #         cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
    #         print("frame is not ready")
    #         # It is better to wait for a while for the next frame to be ready
    #         cv.waitKey(1000)
    #
    #     if cv.waitKey(10) == 27:
    #         break
    #     if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
    #         # If the number of captured frames is equal to the total number of frames,
    #         # we stop
    #         break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    output.release()
