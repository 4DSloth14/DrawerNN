import csv
import json
import math
from pathlib import Path

points_count = 32
target_size = 200

data_folder = Path(__file__).parent
output_file = "spell_training_data.csv"

datasets = {
  "circle": "full_simplified_circle.ndjson",
  "star": "full_simplified_star.ndjson",
  "slash": "full_simplified_line.ndjson",
  "lightning": "full_simplified_lightning.ndjson"
}

max_examples_per_class = 1000


def resample(points, count=32):
  if len(points) < 2 or count < 2:
    return None

  distances = [0.0]
  segment_distances = []
  for i in range(len(points) - 1):
    dx = points[i + 1][0] - points[i][0]
    dy = points[i + 1][1] - points[i][1]
    distance = math.hypot(dx, dy)
    segment_distances.append(distance)
    distances.append(distances[-1] + distance)

  total_length = distances[-1]
  if total_length == 0:
    return None

  result = []
  segment = 0
  for i in range(count):
    target_distance = total_length * i / (count - 1)
    while (
      segment < len(segment_distances) - 1
      and distances[segment + 1] < target_distance
    ):
      segment += 1

    start_distance = distances[segment]
    end_distance = distances[segment + 1]
    if end_distance == start_distance:
      fraction = 0
    else:
      fraction = (target_distance - start_distance) / (end_distance - start_distance)

    x1, y1 = points[segment]
    x2, y2 = points[segment + 1]
    x = x1 + fraction * (x2 - x1)
    y = y1 + fraction * (y2 - y1)
    result.append((x, y))

  return result


def process_drawing(drawing):
  if not isinstance(drawing, list) or len(drawing) != 1:
    return None

  stroke = drawing[0]
  if not isinstance(stroke, list) or len(stroke) != 2:
    return None
  xs, ys = stroke
  if len(xs) != len(ys) or len(xs) < 2:
    return None

  stroke_points = [(x, -y) for x, y in zip(xs, ys)]
  sampled_points = resample(stroke_points, count=points_count)
  if sampled_points is None:
    return None

  min_x = min(x for x, y in sampled_points)
  max_x = max(x for x, y in sampled_points)
  min_y = min(y for x, y in sampled_points)
  max_y = max(y for x, y in sampled_points)

  center_x = (min_x + max_x) / 2
  center_y = (min_y + max_y) / 2
  width = max_x - min_x
  height = max_y - min_y
  longest_side = max(width, height)
  if longest_side == 0:
    return None

  scale = target_size / longest_side
  normalized_points = [
    ((x - center_x) * scale, (y - center_y) * scale)
    for x, y in sampled_points
  ]

  inputs = [x for x, y in normalized_points]
  inputs += [y for x, y in normalized_points]
  return inputs


def main():
  feature_names = (
    [f"x{i}" for i in range(points_count)]
    + [f"y{i}" for i in range(points_count)]
  )
  output_path = data_folder / output_file

  with output_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["label", *feature_names])

    for label, filename in datasets.items():
      dataset_path = data_folder / filename
      examples_written = 0
      with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
          if examples_written >= max_examples_per_class:
            break
          if not line.strip():
            continue

          try:
            record = json.loads(line)
          except json.JSONDecodeError as error:
            raise ValueError(
              f"Invalid JSON in {dataset_path} at line {line_number}"
            ) from error
          if not isinstance(record, dict):
            raise ValueError(
              f"Expected a JSON object in {dataset_path} at line {line_number}"
            )

          inputs = process_drawing(record.get("drawing"))
          if inputs is None:
            continue
          writer.writerow([label, *inputs])
          examples_written += 1


if __name__ == "__main__":
  main()
