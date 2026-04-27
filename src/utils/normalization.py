def min_max_norm(image):
    """
    Normalize the image to the range [0, 1] by subtracting the minimum value and dividing by the range.
    """
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)
