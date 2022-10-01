from utils import *
from copy import deepcopy
import time


# Number of testing points of each group
test_points_num = int(input("Set number of testing points of each group.\n  points : "))

# Load training_data_set from file:
training_data_set = load_data_set()
# Generate test_data_set:
test_data_set = generate_points(test_points_num)

# Show train_data_set visualisation:
graph_mapping_S(100, 7, training_data_set, "*Training data")

k_arr = [1, 3, 7, 15]

# FAST ('F'):
def fast_classification():

    print("\nEffective/relative values [ p -> b |* =>p*4, =>b*b ]:\n"
          " . 10000-5000 points -> 100 blocks\n"
          " . 5000-1000 points -> 50 blocks\n"
          " . 1000-500 points -> 20 blocks\n"
          " . 500-100 points -> 10 blocks\n"
          " . 100-0 points -> 5 blocks\n")


    # Point splitting number ->defining number of blocks when point classifying
    point_split_num = int(input("Set number of blocks, points will be split among.\n   blocks : "))
    # Split data_set into blocks:
    blocked_training_data_set = split_data_set(training_data_set, point_split_num)

    times_arr = []
    scores_arr = []

    for k in k_arr:

        # ---
        start_F = time.time()
        # ---

        # correct_counter:
        cc_F = 0
        blocked_working_data_set = deepcopy(blocked_training_data_set)

        print("\n----------\nK = ", k, ":")

        # Classify all points from testing_data_set:
        for curr_test_point in test_data_set:

            # Parse curr_test_point tuple:
            curr_orig_label = curr_test_point[0]
            curr_test_x = curr_test_point[1]
            curr_test_y = curr_test_point[2]

            # Point classification:
            classified_label_fast = classify_F(curr_test_x, curr_test_y, k, blocked_working_data_set, point_split_num)

            # Correct check:
            if classified_label_fast == curr_orig_label:
                cc_F += 1

        # ---
        end_F = time.time()
        time_F = end_F - start_F
        times_arr.append(time_F)

        score_F = cc_F / len(test_data_set) * 100
        scores_arr.append(score_F)

        reportMsg_F = "[ score: " + str(round(score_F, 2)) + "% | time: " + str(round(time_F, 5)) + "s ]"
        # ---

        # REPORT:
        print(" ->Correctly assigned labels", cc_F, ", out of", len(test_data_set), reportMsg_F)

        # GRAPH:
        graph_mapping_F(100, k, blocked_working_data_set, point_split_num, reportMsg_F)

    # Overall comparison:
    graph_comparison(k_arr, times_arr, "k / time dependence")
    graph_comparison(k_arr, scores_arr, "k / score dependence")


# SLOW ('S')
def slow_classification():

    times_arr = []
    scores_arr = []

    for k in k_arr:

        # ---
        start_S = time.time()
        # ---

        # correct_counters:
        cc_S = 0
        working_data_set = deepcopy(training_data_set)

        print("\n----------\nK = ", k, ":")

        # Classify all points from testing_data_set:
        for curr_test_point in test_data_set:

            # Parse curr_test_point tuple:
            curr_orig_label = curr_test_point[0]
            curr_test_x = curr_test_point[1]
            curr_test_y = curr_test_point[2]

            # Point classification:
            classified_label_S = classify_S(curr_test_x, curr_test_y, k, working_data_set)

            if classified_label_S == curr_orig_label:
                cc_S += 1

        # ---
        end_S = time.time()
        time_S = end_S - start_S
        times_arr.append(time_S)

        score_S = cc_S / len(test_data_set) * 100
        scores_arr.append(score_S)

        reportMsg_S = "[ score: " + str(round(score_S, 2)) + "% | time: " + str(round(time_S, 5)) + "s ]"
        # ---

        # REPORT:
        print(" ->Correctly assigned labels", cc_S, ", out of", len(test_data_set), reportMsg_S)

        # GRAPH:
        graph_mapping_S(100, k, working_data_set, reportMsg_S)


    # Overall comparison:
    graph_comparison(k_arr, times_arr, "k / time dependence")
    graph_comparison(k_arr, scores_arr, "k / score dependence")


while True:

    which = input("Choose classification type -press:\n ->s/S for SLOW\n ->f/F for FAST\n ->c/C to change test_data_set\n ->anything else to END\n")

    if which == "f" or which == "F":
        print("\n -FAST classification-\n")
        fast_classification()

    elif which == "s" or which == "S":
        print("\n -SLOW classification-\n")
        slow_classification()

    elif which == "c" or which == "C":
        test_points_num = int(input("Set number of testing points of each group.\n  points : "))
        test_data_set = generate_points(test_points_num)

    else:
        print("\n END")
        break
