import json

# Function to load the JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Paths to your JSON files
joints2d_path = '/public.hpc/alessandro.folloni2/smpl_study/datasets/FIT3D/band_pull_apart_keypoints.json'

# Load the JSON file
results = load_json(joints2d_path)

# Function to get the "shape" of the JSON in the format "value x value x value"
def get_json_shape(data):
    if isinstance(data, list):
        if len(data) > 0:
            return f"{len(data)} x {get_json_shape(data[0])}"
        else:
            return "0"
    elif isinstance(data, dict):
        return f"{len(data)} keys"
    else:
        return str(type(data).__name__)

# 1. Check the length of the JSON data (frames)
expected_frames = 706  # Adjust this value according to your expectation
actual_frames = len(results)
print(f"Numero totale di elementi nel JSON: {actual_frames}")
if actual_frames != expected_frames:
    print(f"Attenzione: Ci sono {actual_frames - expected_frames} frame in più.")

# 2. Inspect the first and last frame
first_frame = results[0]
last_frame = results[-1]

print("\n--- Primo Frame ---")
print(f"Shape: {get_json_shape(first_frame)}")
print(first_frame)

print("\n--- Ultimo Frame ---")
print(f"Shape: {get_json_shape(last_frame)}")
print(last_frame)

# 3. Identify duplicate frames
duplicate_frames = []
for i in range(1, len(results)):
    if results[i] == results[i-1]:
        duplicate_frames.append(i)

if duplicate_frames:
    print(f"\nFrame duplicati trovati agli indici: {duplicate_frames}")
else:
    print("\nNessun frame duplicato trovato.")


print("\n--- Struttura Generale dei Frame ---")
shapes = set()
for frame in results:
    shape = get_json_shape(frame)
    shapes.add(shape)

print(f"Diversi formati di frame trovati: {shapes}")

# If all frames have the same structure, shapes should have only one element
if len(shapes) == 1:
    print("Tutti i frame hanno la stessa struttura.")
else:
    print("Sono stati trovati frame con strutture diverse.")

# Optional: Remove duplicate or empty frames
# Removing duplicate frames
results_cleaned = []
previous_frame = None
for frame in results:
    if frame != previous_frame:
        results_cleaned.append(frame)
    previous_frame = frame
