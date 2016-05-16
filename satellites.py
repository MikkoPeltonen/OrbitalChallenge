"""
Mikko Peltonen 2016

Finds the shortest path from point A to point B via randomly placed satellites. By default the program searches for satellites
in a file named satellites.txt, located in the same directory as the script.

Operation:

1. Earth is perfectly round with a radius of 6371 km.
2. Geographical coordinates (latitude, longitude, altitude) are converted to Cartesian coordinates using the following formulas:
   x = r * cos(latitude) * cos(longitude)
   y = r * cos(latitude) * sin(longitude)
   z = r * sin(latitude)
   where x, y and z are the Cartesian coordinates and r = R + altitude, in which R is the radius of Earth.
3. Points A and B are on the surface of Earth (altitude 0).
4. Two satellites can communicate if they have a direct line of sight. Line of sight is determined by whether the line segment
   connecting the satellites is no closer than the radius of Earth to the Cartesian coordinate origin. This is calculated by
   |(O - S1) × (O - S2)| / |S2 - S1| where O is origin (0, 0, 0), S1 and S2 are the points where the satellites are located
   at, |v| denotes the length of the vector v and × is vector cross product. There is a maximum number of connections, which
   is nCr(20, 2) = 190. All 190 combinations are tested.
   Reasoning: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
5. A satellite can communicate with a point on Earth surface if they have a straight line of sight. Line of sight between a
   satellite and a point on Earth is determined by whether the angle between vectors AO (from point A on the surface of Earth
   to origin O) and AS (from point A to satellite S) is greater than or equal to π/2 (90 degrees).
6. Knowing all the possible connections between the satellites and the points on Earth, and the distance of each connection, we
   compute the shortest distance from point A to point B using Dijkstra's algorithm. More details can be found here:
   https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

For visualization the program utilizes matplotlib. Each point is drawn around the spherical Earth and possible connections
are indicated with a green line. Blue lines indicate all available connections from the surface of Earth to satellites and
red lines indicate the shortest path.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import matplotlib.colors as colors
from collections import OrderedDict, defaultdict
from mpl_toolkits.mplot3d import Axes3D, proj3d

center = start = end = np.array([0, 0, 0])
radius = 6371
satellites = OrderedDict()

seed = ''
source = 'satellites.txt'


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, node):
        self.nodes.add(node)

    def get_nodes(self):
        return self.nodes

    def add_edge(self, from_node, to_node, distance):
        """ Add connection between two nodes """
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance


def dijkstra(graph, origin, destination):
    """ Determine the shortest path from the source node to the destination node and return the path """
    visited_nodes = {origin: 0}
    unvisited_nodes = graph.get_nodes()
    previous = {}

    for node in unvisited_nodes:
        previous[node] = None

    while unvisited_nodes:
        # min_node is the node with least distance. Origin node will be selected first.
        min_node = None
        for node in unvisited_nodes:
            if node in visited_nodes:
                if min_node is None or visited_nodes[node] < visited_nodes[min_node]:
                    min_node = node

        # If min_node is still None, there is no path from source to destination
        if min_node is None:
            break

        # Found destination, return the path in reverse (to make it go from source to destination)
        if min_node == destination:
            nodes = []
            node = min_node
            while node in previous:
                nodes.insert(0, node)
                node = previous.get(node)

            return nodes

        unvisited_nodes.remove(min_node)
        current_distance = visited_nodes[min_node]

        # Check all nodes connected to min_node
        for neighbour in graph.edges[min_node]:
            try:
                distance = current_distance + graph.distances[(min_node, neighbour)]
            except:
                continue

            # If needed, update the node's distance
            if neighbour not in visited_nodes or distance < visited_nodes[neighbour]:
                visited_nodes[neighbour] = distance
                previous[neighbour] = min_node

    # No path was found
    return None


def unit_vector(vector):
    """ Return unit vector of the given vector """
    return vector / np.linalg.norm(vector)


def angle_between(vector1, vector2):
    """ Return the angle between two vectors """
    unit_vector1 = unit_vector(vector1)
    unit_vector2 = unit_vector(vector2)
    
    return np.arccos(np.clip(np.dot(unit_vector1, unit_vector2), -1.0, 1.0))


def to_cartesian(latitude, longitude, altitude):
    """ Convert geographic coordinates to Cartesian """
    r = radius + altitude
    x = r * np.cos(math.radians(latitude)) * np.cos(math.radians(longitude))
    y = r * np.cos(math.radians(latitude)) * np.sin(math.radians(longitude))
    z = r * np.sin(math.radians(latitude))

    return np.array([x, y, z])


def distance_between(satellite1, satellite2):
    """ Return distance between two satellites """
    return np.linalg.norm(satellite1 - satellite2)


def can_connect(satellite1, satellite2):
    """ Return True if the satellites have straight line of sight """
    distance = np.linalg.norm(np.cross(center - satellite1, center - satellite2)) / np.linalg.norm(satellite2 - satellite1)
    return distance > radius

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')

# Draw sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='w', zorder=-1)

graph = Graph()

# Process satellite source file
with open(source) as satellites_file:
    for i, line in enumerate(satellites_file):
        data = line.split(',')
        if i == 0:
            seed = line[7:].strip()
        elif i == 21:
            # Last line contains route details
            start = to_cartesian(float(data[1]), float(data[2]), 0.0)
            end = to_cartesian(float(data[3]), float(data[4]), 0.0)
            graph.add_node('START')
            graph.add_node('END')
        else:
            # Other lines have satellite data
            satellites[data[0]] = to_cartesian(float(data[1]), float(data[2]), float(data[3]))
            graph.add_node(data[0])

color_names = list(colors.cnames.keys())

# Draw satellite points
for satellite, coordinates in satellites.items():
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], c=color_names.pop(1), s=100, label=satellite)
    ax.text(coordinates[0], coordinates[1], coordinates[2], satellite)

# Draw start and end points
ax.scatter(start[0], start[1], start[2], c='m', s=100)
ax.scatter(end[0], end[1], end[2], c='y', s=100)
ax.text(start[0], start[1], start[2], 'START')
ax.text(end[0], end[1], end[2], 'END')

# Compute all possible inter-satellite connections
for satellite_pair in itertools.combinations(satellites, 2):
    if can_connect(satellites[satellite_pair[0]], satellites[satellite_pair[1]]):
        # Draw connection line
        satellite1_coords = satellites[satellite_pair[0]]
        satellite2_coords = satellites[satellite_pair[1]]
        ax.plot([satellite1_coords[0], satellite2_coords[0]],
                [satellite1_coords[1], satellite2_coords[1]],
                [satellite1_coords[2], satellite2_coords[2]], c='g')

        graph.add_edge(satellite_pair[0], satellite_pair[1], distance_between(satellite1_coords, satellite2_coords))

# Compute all possible connections from satellites to Earth
for satellite, coordinates in satellites.items():
    if angle_between(coordinates - start, center - start) >= np.pi / 2:
        # Satellite can connect to start point
        ax.plot([coordinates[0], start[0]], [coordinates[1], start[1]], [coordinates[2], start[2]], c='b')
        graph.add_edge(satellite, 'START', distance_between(coordinates, start))

    if angle_between(coordinates - end, center - end) >= np.pi / 2:
        # Satellite can connect to end point
        ax.plot([coordinates[0], end[0]], [coordinates[1], end[1]], [coordinates[2], end[2]], c='b')
        graph.add_edge(satellite, 'END', distance_between(coordinates, end))

# Show scatter plot legends
plt.legend(loc='lower left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

# Calculate shortest path from start to end
path = dijkstra(graph, 'START', 'END')

if not path:
    print('No path found between the given points with seed', seed)
    quit()

print('Seed:         ', seed)
print('Shortest path:', ','.join(path[1:-1]))

# Draw shortest path
xcoords = [satellites[step][0] for step in path[1:-1]]
xcoords.insert(0, start[0])
xcoords.append(end[0])

ycoords = [satellites[step][1] for step in path[1:-1]]
ycoords.insert(0, start[1])
ycoords.append(end[1])

zcoords = [satellites[step][2] for step in path[1:-1]]
zcoords.insert(0, start[2])
zcoords.append(end[2])

ax.plot(xcoords, ycoords, zcoords, c='r')

# Show graph window maximized
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()