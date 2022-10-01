import random
import matplotlib.pyplot as plt
from collections import Counter

def get_block_size(parts_n):
    return 10000/parts_n


def load_data_set():
    points_arr = []
    curr_arr = []

    with open("1.txt", "r") as file:

        for line_str in file:

            if (line_str[0] >= 'A') and (line_str[0] <= 'Z'):
                if len(curr_arr) > 0:
                    points_arr.append(curr_arr)
                curr_arr = []
            else:
                line_arr = line_str.split()
                x = int(line_arr[0])
                y = int(line_arr[1])
                curr_arr.append((x, y))

    points_arr.append(curr_arr)
    return points_arr


def generate_points(n):
    generated_points = []

    # Generate n points from each group:
    for i in range(n):
        orig_label, x, y = get_Rpoint()
        generated_points.append((orig_label, x, y))

        orig_label, x, y = get_Gpoint()
        generated_points.append((orig_label, x, y))

        orig_label, x, y = get_Bpoint()
        generated_points.append((orig_label, x, y))

        orig_label, x, y = get_Ppoint()
        generated_points.append((orig_label, x, y))

    return generated_points


def generate_randomly():
    return random.randint(-5000, 5000), random.randint(-5000, 5000)


# Generate point of 0.label/group = R:
def get_Rpoint():
    if random.random() <= 0.99:
        x = random.randint(-5000, 499)
        y = random.randint(-5000, 499)
    else:
        x, y = generate_randomly()
    return 0, x, y

# Generate point of 1.label/group = G:
def get_Gpoint():
    if random.random() <= 0.99:
        x = random.randint(-499, 5000)
        y = random.randint(-5000, 499)
    else:
        x, y = generate_randomly()
    return 1, x, y

# Generate point of 2.label/group = B:
def get_Bpoint():
    if random.random() <= 0.99:
        x = random.randint(-5000, 499)
        y = random.randint(-499, 5000)
    else:
        x, y = generate_randomly()
    return 2, x, y

# Generate point of 3.label/group = P:
def get_Ppoint():
    if random.random() <= 0.99:
        x = random.randint(-499, 5000)
        y = random.randint(-499, 5000)
    else:
        x, y = generate_randomly()
    return 3, x, y


# Split data_set into n blocks according to set number of blocks:
def split_data_set(data_set, point_split_num):
    block_size = get_block_size(point_split_num)

    # Create split/blocked data_set list:
    split_set = []
    for i in range(point_split_num):
        split_set.append([])
        for _ in range(point_split_num):
            split_set[i].append([])

    # Distribute existing points within the blocks according to their coordinates:
    for label, label_arr in enumerate(data_set):
        for point in label_arr:

            # Parse point tuple:
            x = point[0]
            y = point[1]

            # Get their blocks index to be append to according to their coordinates:
            x_index = (x+5000) // block_size
            y_index = (y+5000) // block_size

            # osetrenie 5000-cky ->indexuje uz na neexist. poz.
            if x_index == point_split_num:
                x_index -= 1
            if y_index == point_split_num:
                y_index -= 1

            # Append the point tuple to its block:
            split_set[int(y_index)][int(x_index)].append((label, x, y))

    return split_set


# Function that returns at least 'k' nearest neighbors:
def get_neighbors_F(working_data_set, test_x, test_y, parts_n, k):

    neighbors = []
    blocks_to_check = get_blocks_to_check(working_data_set, test_x, test_y, parts_n, k)

    # For each point in the data_set:
    for curr_train_block in blocks_to_check:

        for curr_train_point in curr_train_block:

            # Parse curr_train_point tuple
            curr_orig_label = curr_train_point[0]
            curr_train_x = curr_train_point[1]
            curr_train_y = curr_train_point[2]

            # Calculate  distance:
            distance = euclidean_distance(test_x, test_y, curr_train_x, curr_train_y)
            # Append distance and label of the measured point to the list of distances:
            neighbors.append((distance, curr_orig_label))

    return neighbors

def get_neighbors_S(working_data_set, test_x, test_y):

    neighbors = []

    # For each point in the data_set:
    for index, label_arr in enumerate(working_data_set):

        for curr_train_point in label_arr:

            # Parse curr_train_point tuple
            # curr_orig_label = curr_train_point[0]
            curr_train_x = curr_train_point[0]
            curr_train_y = curr_train_point[1]

            # Calculate  distance:
            distance = euclidean_distance(test_x, test_y, curr_train_x, curr_train_y)
            # Append distance and label of the measured point to the list of distances:
            neighbors.append((distance, index))

    return neighbors


# Function that returns required blocks containing at least 'k' nearest neighbors/points:
def get_blocks_to_check(working_data_set, test_x, test_y, parts_n, k):
    block_size = get_block_size(parts_n)

    x_index = (test_x + 5000) // block_size
    y_index = (test_y + 5000) // block_size

    # osetrenie 5000-cky ->indexuje uz na neexist. poz.
    if x_index == parts_n:
        x_index -= 1
    if y_index == parts_n:
        y_index -= 1

    mid_dst = 1
    corner_shift = 0
    while True:

        blocks_to_check = []
        num_of_points = 0
        mid_dst += 2
        corner_shift += 1

        corner_x = int(x_index) - corner_shift
        corner_y = int(y_index) - corner_shift

        for i in range(mid_dst):
            for j in range(mid_dst):
                curr_y_index = corner_y + i
                curr_x_index = corner_x + j

                if (0 <= curr_x_index < len(working_data_set[0])) and (0 <= curr_y_index < len(working_data_set)):
                    blocks_to_check.append(working_data_set[int(curr_y_index)][int(curr_x_index)])
                    num_of_points += len(working_data_set[int(curr_y_index)][int(curr_x_index)])

        if num_of_points >= k:
            break

    return blocks_to_check


# Calculate and return distance between 2 points:
def euclidean_distance(x1, y1, x2, y2):
    x_dst = abs(x1-x2)
    y_dst = abs(y1-y2)

    # c^2 = a^2 + b^2
    distance = (x_dst**2 + y_dst**2)**0.5

    return distance


# Function that returns most frequent label within 'k' nearest neighbors:
# (If more "max" labels ->choose the closest one)
def get_classified_label(k_nearest_labels):
    counted_labels = Counter(k_nearest_labels)

    maxValue = max(counted_labels.values())
    all_max_labels = [k for k, v in counted_labels.items() if v == maxValue]

    # If more than 1 max label ->return the closest one:
    if len(all_max_labels) > 1:
        for label in k_nearest_labels:
            if label in all_max_labels:
                return label
    else:
        return all_max_labels[0]


def classify_F(x, y, k, working_data_set, split_blocks_num):

    # Neighbor distances and labels:
    neighbors = get_neighbors_F(working_data_set, x, y, split_blocks_num, k)

    # Sort neighbor distances and labels:
    sorted_neighbors = sorted(neighbors)

    # Get the first k labels from sorted_neighbors list of distances and labels:
    k_nearest_labels = [sorted_neighbors[i][1] for i in range(k)]

    # Get most frequent label = classified label:
    classified_label = get_classified_label(k_nearest_labels)

    # Get block position of the new test_point:
    curr_x_index = (x+5000) // get_block_size(split_blocks_num)
    curr_y_index = (y+5000) // get_block_size(split_blocks_num)

    # Osetrenie 5000-cky ->indexuje uz na neexist. poz.
    if curr_x_index == split_blocks_num:
        curr_x_index -= 1
    if curr_y_index == split_blocks_num:
        curr_y_index -= 1

    # Add the point to his block:
    working_data_set[int(curr_y_index)][int(curr_x_index)].append((classified_label, x, y))

    return classified_label

def classify_S(x, y, k, working_data_set):

    # Neighbor distances and labels:
    neighbors = get_neighbors_S(working_data_set, x, y)

    # Sort neighbor distances and labels:
    sorted_neighbors = sorted(neighbors)

    # Get the first k labels from sorted_neighbors list of distances and labels:
    k_nearest_labels = [sorted_neighbors[i][1] for i in range(k)]

    # Get most frequent label = classified label:
    classified_label = get_classified_label(k_nearest_labels)

    # Add the point to his block:
    working_data_set[classified_label].append((x, y))

    return classified_label


def graph_mapping_F(graph_blocks_num, k, working_data_set, split_blocks_num, msg):
    plt.rcParams["figure.figsize"] = (12, 6)
    coloring = ["red", "green", "blue", "purple"]
    plt.suptitle("K nearest neighbors : k = " + str(k) + "\n" + msg)

    # Subplot-1 (points):
    colors = []
    x = []
    y = []
    for line in working_data_set:
        for block in line:
            for point_tuple in block:
                colors.append(coloring[point_tuple[0]])
                x.append(point_tuple[1])
                y.append(point_tuple[2])

    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color=colors, marker=".", s=10)

    # Subplot-2 (maps):
    colors = []
    x = []
    y = []
    block_size = get_block_size(graph_blocks_num)

    for i in range(graph_blocks_num):
        mid_point_y = i*block_size + block_size//2 - 5000
        for j in range(graph_blocks_num):
            mid_point_x = j*block_size + block_size//2 - 5000

            neighbors = get_neighbors_F(working_data_set, mid_point_x, mid_point_y, split_blocks_num, k)
            sorted_neighbors = sorted(neighbors)
            k_nearest_labels = [sorted_neighbors[i][1] for i in range(k)]
            colors.append(coloring[get_classified_label(k_nearest_labels)])

            x.append(mid_point_x)
            y.append(mid_point_y)

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color=colors, marker="h", s=block_size, alpha=0.5)

    plt.show()

def graph_mapping_S(graph_blocks_num, k, working_data_set, msg):
    plt.rcParams["figure.figsize"] = (12, 6)
    coloring = ["red", "green", "blue", "purple"]
    plt.suptitle("K nearest neighbors : k = " + str(k) + "\n" + msg)

    # Subplot-1 (points):
    colors = []
    x = []
    y = []
    for index, label_arr in enumerate(working_data_set):
        for point_tuple in label_arr:
            colors.append(coloring[index])
            x.append(point_tuple[0])
            y.append(point_tuple[1])

    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color=colors, marker=".", s=10)

    # Subplot-2 (maps):
    colors = []
    x = []
    y = []
    block_size = get_block_size(graph_blocks_num)

    for i in range(graph_blocks_num):
        mid_point_y = i*block_size + block_size//2 - 5000
        for j in range(graph_blocks_num):
            mid_point_x = j*block_size + block_size//2 - 5000

            neighbors = get_neighbors_S(working_data_set, mid_point_x, mid_point_y)
            sorted_neighbors = sorted(neighbors)
            k_nearest_labels = [sorted_neighbors[i][1] for i in range(k)]
            colors.append(coloring[get_classified_label(k_nearest_labels)])

            x.append(mid_point_x)
            y.append(mid_point_y)

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color=colors, marker="h", s=block_size, alpha=0.5)

    plt.show()


def graph_comparison(ks, others, title):
    print("\nOverall " + title + ":")
    for i, t in enumerate(others):
        print(" k =", ks[i], ":", round(t, 4))
    print()

    plt.rcParams["figure.figsize"] = (8, 4)
    plt.title(title)
    plt.xlabel("K-value")
    plt.plot(ks, others, "h:b", ms=8)
    plt.show()