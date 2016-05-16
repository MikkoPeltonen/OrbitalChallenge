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
