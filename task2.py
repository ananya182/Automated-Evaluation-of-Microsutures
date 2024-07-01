import math

# Sample dictionary with x, y coordinates
coordinates_dict = {'point1': (1, 2), 'point2': (4, 6), 'point3': (7, 8)}

# Function to calculate distance between two points
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate distances between consecutive points
def inter_suture_distance(coordinates_dict):
    distances = [distance(coordinates_dict[key], coordinates_dict[next_key]) for key, next_key in zip(coordinates_dict, list(coordinates_dict.keys())[1:])]

    # Calculate mean and variance
    mean_distance = sum(distances) / len(distances)

    # Calculate variance
    variance_distance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)

    return mean_distance, variance_distance
