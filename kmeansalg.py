import platform
import random
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt

plt.show()
plt.grid(True)
plt.ylabel('"Changes of centroids"')
from mlxtend.data import loadlocal_mnist

if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
else:
    X, y = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')
    test_x, test_y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')  # Files upload


def create_centroid():  # יוצר סנטרואיד בודד
    m = []  # 28x28 dimension
    for i in range(0, 784, 1):
        m.append(random.uniform(0, 1))
    return m


def create_centroids(k):  # יוצר k סנטרואידים
    centroids = []
    for i in range(0, k, 1):
        centroid = create_centroid()
        centroids.append(centroid)
    centroids = np.asarray(centroids)
    return centroids


def how_differ(centroids, previous_centroids):
    centroids = np.asarray(centroids)
    previous_centroids = np.asarray(previous_centroids)
    ans = abs(centroids - previous_centroids)
    errors = []
    for i in ans:
        for j in i:
            errors.append(j)
    return errors


def close_enough(centroids, previous_centroids, gap):
    norm1 = np.linalg.norm(centroids)
    norm2 = np.linalg.norm(previous_centroids)
    if gap == 0:
        return abs(norm1 - norm2) == gap
    return abs(norm1 - norm2) < gap  # gap = 0.01  -best yet


def create_sets(points, centroids_indices):
    sets = []
    for j in range(0, 10, 1):
        sets.append([])
    for i in range(0, len(points), 1):
        # sets[centroids_indices[i]].append(points[i])  // using the closest_centroid func
        sets[centroids_indices[i]].append(points[i])
    return sets


def find_closest_centroids(images, centroids):
    """
    Returns the closest centroids in idx for a dataset images
    where each row is a single example."""""
    idx = np.zeros((images.shape[0], 1), dtype=np.int8)
    for i in range(images.shape[0]):
        distances = np.linalg.norm(images[i] - centroids, axis=1)
        min_dst = np.argmin(distances)
        idx[i] = min_dst
    output = []
    for i in range(0, len(idx), 1):
        output.append(idx[i][0])
    return output


def define_sets(points, centroids):
    # centroids_indices = closest_centroid((points), np.array(centroids))
    centroids_indices = find_closest_centroids(points, centroids)
    sets = create_sets(points, centroids_indices)
    return sets


def define_centroids(sets, cents):
    centroids = []
    for i in range(0, len(sets), 1):
        arr = np.asarray(sets[i])
        if len(arr) == 0:
            print("Bad centroids choice!! RESTARTING...\n\n")
            main()
            sys.exit()
        mean = np.mean(arr, axis=0)
        centroids.append(mean)
    centroids = np.asarray(centroids)
    return centroids


def most_common(sets, centroids, indices):
    hist = []
    for i in range(0, 10, 1):
        hist.append([0] * 10)
    hist = np.asarray(hist)
    for x in range(0, len(X), 1):
        label = y[x]
        belongs_to_cent_ind = indices[x]
        hist[belongs_to_cent_ind][label] = hist[belongs_to_cent_ind][label] + 1

    # ct = 0
    # for s in range(0, len(sets), 1):
    #     print("the centroid in index", s, "have:", len(sets[s]), "images")
    #     ct = ct + len(sets[s])

    def find_index(arr, maximum):
        for j in range(0, 10, 1):
            if arr[j] == maximum:
                return j

    def who_is_missing(represents):
        missing = []
        for i in range(0, len(represents), 1):
            if not represents.__contains__(i):
                missing.append(i)
        return missing

    represents = []
    for i in range(0, 10, 1):
        maxi = np.max(hist[i])
        ind = find_index(hist[i], maxi)
        represents.append(ind)
    missing = who_is_missing(represents)
    return represents, missing, hist


def results(images, centroids, sets):
    """"return: success - number of success
                represents - our module array, where centroid(=index) gets a label
                missing - the images that are missing
                hist - a histogram table"""
    indices = find_closest_centroids(images, centroids)
    # indices = closest_centroid(images, centroids)
    represents, missing, hist = most_common(sets, centroids, indices)
    indexs = find_closest_centroids(test_x, centroids)
    success = 0
    for i in range(0, len(test_x), 1):
        if test_y[i] == represents[indexs[i]]:
            success = success + 1
    return success, represents, missing, hist


def print_to_graph(new, old, t, downs, ups):
    if new < old:
        print("went down by", old - new)
        downs = downs + 1
    else:
        print("INCREASED by", new - old)
        ups = ups + 1
    print("to", new)
    if abs(new - old) < 1.0:
        if new < old:
            plt.scatter([t], [new], c='green')
        else:
            plt.scatter([t], [new], c='red')
    return downs, ups


def main():
    start = timeit.default_timer()

    images = X / 255
    centroids = create_centroids(10)
    previous_centroids = centroids
    downs, ups, indices, sets = 0, 0, None, None

    mm = 0
    t = 1
    while (t == 1) or (not close_enough(centroids, previous_centroids, gap=0.0034)):  # 0.0034
        print("\nprocessing iteration number", t, "...")
        old = np.linalg.norm(np.linalg.norm(centroids) - np.linalg.norm(previous_centroids))
        previous_centroids = centroids
        sets = define_sets(images, centroids)
        centroids = define_centroids(sets, centroids)
        new = np.linalg.norm(np.linalg.norm(centroids) - np.linalg.norm(previous_centroids))
        downs, ups = print_to_graph(new, old, t, downs, ups)
        t = t + 1
    #   LOOP-END----------------------------------------------------------
    stop = timeit.default_timer()
    time = stop - start
    minutes = int(time / 60)
    seconds = int(time % 60)
    print("\nFinished after ", t - 1, " rounds!")
    print("the \"biggest change\" is: ")
    print(max(how_differ(centroids, previous_centroids)))
    print("Went up", ups, "times and went down", downs, "times.")
    print("Total runtime: ", minutes, "minutes and", seconds, "seconds.\n")

    success, represents, missing, hist = results(images, centroids, sets)
    print("\n The histogram table:\n", hist)
    print("\nThe module should represent this numbers:\n", represents)
    print("the numbers", missing, "are missing!")
    print("\n\n", (success / len(test_y)) * 100, "% success!!!")
    plt.xlabel(((success / len(test_y)) * 100))
    plt.show()


main()


# def wise_choice(k):
#     # centroids = wise_choice(int(input("how much data for each centroid? (max- 5420)  ")))
#     ct = [[], [], [], [], [], [], [], [], [], []]
#     ind = 0
#     while ind < 10:
#         for j in range(0, k, 1):
#             ct[ind].append(c[ind].pop())
#         ind = ind + 1
#
#     out = []
#     for m in range(0, 10, 1):
#         mean = np.mean(ct[m], axis=0)
#         out.append(mean)
#     return np.asarray(out)



    # c = [[], [], [], [], [], [], [], [], [], [], ]
    # i = 0
    # images = X / 255
    # while i < 60000:
    #     if y[i] == 0:
    #         c[0].append(images[0])
    #     elif y[i] == 1:
    #         c[1].append(images[1])
    #     elif y[i] == 2:
    #         c[2].append(images[2])
    #     elif y[i] == 3:
    #         c[3].append(images[3])
    #     elif y[i] == 4:
    #         c[4].append(images[4])
    #     elif y[i] == 5:
    #         c[5].append(images[5])
    #     elif y[i] == 6:
    #         c[6].append(images[6])
    #     elif y[i] == 7:
    #         c[7].append(images[7])
    #     elif y[i] == 8:
    #         c[8].append(images[8])
    #     elif y[i] == 9:
    #         c[9].append(images[9])
    #     i = i + 1  # my try!!!



# check()

# result = None
# while result is None:
#     try:
#         # connect
#         result = main()
#     except:
#          pass
# indexs = closest_centroid(test_x, centroids)
#     dup = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     for i in range(0, len(represents), 1):
#         dup[represents[i]] = dup[represents[i]] + 1
#     j = 0
#     for i in range(0, len(dup), 1):
#         if dup[i] == 2:
#             for j in range(0, len(represents), 1):
#                 if represents[j] == i:
#                     represents[j] = missing.pop()
#                     break
#     success = 0
#     for i in range(0, len(test_x), 1):
#         if test_y[i] == represents[indexs[i]]:
#             success = success + 1
#     print("\n", represents)
#     print("well?", ((success / len(test_y)) * 100))
