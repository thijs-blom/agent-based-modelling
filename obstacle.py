class Obstacle():
    """Base class for an obstacle that agents must avoid"""
    def get_closest_point(self, other_point):
        """Compute the closest point of the obstacle to the given point"""
        raise NotImplementedError
