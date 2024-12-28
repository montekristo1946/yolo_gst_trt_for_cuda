#include <iostream>
#include <vector>
using namespace std;

struct Point {
    // Coordinates of the point
    double x, y;
};

// Function to check if a point is inside a polygon using
// the ray-casting algorithm
bool isPointInPolygon(const vector<Point>& polygon,
                      const Point& point)
{
    // Number of vertices in the polygon
    int n = polygon.size();
    // Count of intersections
    int count = 0;

    // Iterate through each edge of the polygon
    for (int i = 0; i < n; i++) {
        Point p1 = polygon[i];
        // Ensure the last point connects to the first point
        Point p2 = polygon[(i + 1) % n];

        // Check if the point's y-coordinate is within the
        // edge's y-range and if the point is to the left of
        // the edge
        if ((point.y > min(p1.y, p2.y))
            && (point.y <= max(p1.y, p2.y))
            && (point.x <= max(p1.x, p2.x))) {
            // Calculate the x-coordinate of the
            // intersection of the edge with a horizontal
            // line through the point
            double xIntersect = (point.y - p1.y)
                                    * (p2.x - p1.x)
                                    / (p2.y - p1.y)
                                + p1.x;
            // If the edge is vertical or the point's
            // x-coordinate is less than or equal to the
            // intersection x-coordinate, increment count
            if (p1.x == p2.x || point.x <= xIntersect) {
                count++;
            }
        }
    }
    // If the number of intersections is odd, the point is
    // inside the polygon
    return count % 2 == 1;
}

int main()
{
    vector<Point> polygon
        = { { 1, 1 }, { 1, 5 }, { 5, 5 }, { 5, 1 } };
    // Define a polygon
    Point point = { 3, 3 }; // Define a point to check

    // Check if the point is inside the polygon and print
    // the result
    if (isPointInPolygon(polygon, point)) {
        cout << "Point is inside the polygon." << endl;
    }
    else {
        cout << "Point is outside the polygon." << endl;
    }

    return 0;
}
