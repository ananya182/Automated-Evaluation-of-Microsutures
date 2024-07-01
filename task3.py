import math

def calculate_angle(point1, point2):
    y1, x1 = point1
    y2, x2 = point2

    y1*=-1
    y2*=-1
    
    # Calculate the angle with respect to the x-axis
    angle_radians = math.atan((y2 - y1)/(x2 - x1))
    
    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def find_angle_mean_var(dict1,dict2):

    # Lists to store angles for mean and variance calculations
    angles = []

    # Iterate over keys and calculate angles
    for key in dict1.keys():
        point_dict1 = dict1[key]
        point_dict2 = dict2[key]
        
        # Calculate the angle between the corresponding points and the x-axis
        angle = calculate_angle(point_dict1, point_dict2)
        
        angles.append(angle)
        
        # print(f"Angle between {key} points and the x-axis: {angle} degrees")

    # Calculate mean and variance of all angles
    mean_angle = sum(angles) / len(angles)
    variance_angle = sum((a - mean_angle) ** 2 for a in angles) / len(angles)

    return mean_angle, variance_angle
