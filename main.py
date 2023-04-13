import numpy as np
import pandas as pd

column_names = [
    'elevation',
    'aspect',
    'slope',
    'horizontal_distance_hydrology',
    'vertical_distance_hydrology',
    'horizontal_distance_roadways',
    'hillshade_9am',
    'hillshade_Noon',
    'hillshade_3pm',
    'horizontal_distance_to_fire_points',
    'wilderness_1',
    'wilderness_2',
    'wilderness_3',
    'wilderness_4',
    'soil_1',
    'soil_2',
    'soil_3',
    'soil_4',
    'soil_5',
    'soil_6',
    'soil_7',
    'soil_8',
    'soil_9',
    'soil_10',
    'soil_11',
    'soil_12',
    'soil_13',
    'soil_14',
    'soil_15',
    'soil_16',
    'soil_17',
    'soil_18',
    'soil_19',
    'soil_20',
    'soil_21',
    'soil_22',
    'soil_23',
    'soil_24',
    'soil_25',
    'soil_26',
    'soil_27',
    'soil_28',
    'soil_29',
    'soil_30',
    'soil_31',
    'soil_32',
    'soil_33',
    'soil_34',
    'soil_35',
    'soil_36',
    'soil_37',
    'soil_38',
    'soil_39',
    'soil_40',
    'cover_type'
]

df = pd.read_csv('./covtype.data',
                 header=None,
                 names=column_names
                 )

cover_types_unique = df['cover_type'].nunique()


# for 1 task
def get_cut_points(data, number_of_points):
    split_data = np.array_split(np.sort(data), number_of_points)
    split_points = [points[-1] for points in split_data]

    return split_points


cut_points = get_cut_points(df['elevation'], cover_types_unique)


def search_insert_position(data, element):
    """
    returns the index where a given element should be put into a sorted array
    works for O(log n)
    """
    left, right = 0, len(data) - 1

    while left <= right:
        mid = (left + right) // 2

        if data[mid] == element:
            return mid
        elif data[mid] < element:
            left = mid + 1
        else:
            right = mid - 1

    return left


def get_cover_type_by_elevation(row):
    elevation = row['elevation']
    idx = search_insert_position(cut_points, elevation)

    return idx + 1
# -------------------------------------------------------------------------------------


# Task 2.
# Implement a very simple heuristic that will classify the data (It doesn't need to be accurate)
#
# - The implemented heuristic is based on the elevation column.
# We divide elevation values into equal parts the amount of which is equal to the quantity of cover types (7)
# and according to it bringing up the class
#

df['elevation_heuristic_cover_type'] = df.apply(get_cover_type_by_elevation, axis=1)
print(np.sum(df['elevation_heuristic_cover_type'] == df['cover_type']) / len(df), '- the portion of correct answers')


