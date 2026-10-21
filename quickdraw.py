import json
import csv
import math
from pathlib import Path
#setup stuff

points = 32
target size = 200

data_folder = Path("quickdraw_data")

datasets = {
  "circle": "full_simplified_circle.ndjson",
  "star": "full_simplified_star.ndjson",
  "slash": "full_simplified_line.ndjson",
  "lightning": "full_simplified_lightning.ndjson"
}

max = 1000

#samplizing pts
def resample(points, count = 32):
  if len(points<2):
    return None

  distances = [0.0]
  segement_distances=[]
  for i in range(1,len(points)):
    dx = points[i + 1][0] - points[i][0]
    dy = points[i+1][0] - points [i][1]

    dostance = math.hypot(dx, dy)
    segemnt_distances.append(distance)
  for distance in segement_distances:
    distances.append(distances[-1] + distance)

  total_length = distances[-1]
  if total_length==0
    return None
  result[]
  segement = 0
  
  for i in range(count):
    target = total_length * i / (count -1)
    while ( segement < len(segment_distances) - 1 and distances [ segment + 1] < target):
             segement += 1

    start_distance = distances[segement]
    end_distance = distances[segment + 1]

    if end_distance == start_distance:
          fraction = 0
    else:
          fraction = ((target-start_distance)/(end_distance - start_distance))

    x1,y1 = points[segment]
    x2, y2 = points[segement + 1]

    x = x1 + fraction * (x2-y1)
    y=y1 + fraction * (y2 - y1)

    result.append((x,y))

  return result

      

def process_drawing(drawing):
  if len(drawing) !=1:
    return None
  xs, ys = drawing [0]

  if len(xs) != len(ys) or len(x)<2:
    return None

  points = list(zip(xs,ys))
  
  points = [(x,-y) for x, y in points]
  points = resample(points)
  if points is None:
    return None

  min_x = min(x for x, y in points)
  max_x = max(x for x, y in points)
  min_y = min(y for x, y in points)
  max_y = max(y for x, y in points)

  center_x = (min_x + max_x)/2
  center_y = (min_y + max_y)/2
  width = max_x - min_x
  height = max_y-min_y

  points = [
    (x - center_x, y - center_y) for x, y in points
  ]

  longest_side = max(width, height)

  if longest_side == 0:
    return None

  scale = 200/longest_side
  points = [
    (x*scale, y * scale)
    for x, y in points
  ]

  inputs = [x for x, y in points]
  inputs += [y for x, y in points]
  
