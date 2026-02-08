"""Quick investigation script to determine what LanceDB's _distance returns."""
import tempfile
import numpy as np
from pathlib import Path

# Create test vectors
def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# Test with known vectors
vec1 = normalize(np.array([1.0, 0.0, 0.0]))
vec2 = normalize(np.array([0.9, 0.1, 0.0]))  # Very similar to vec1
vec3 = normalize(np.array([0.0, 1.0, 0.0]))  # Orthogonal to vec1

print("Test vectors (normalized):")
print(f"vec1: {vec1}, norm: {np.linalg.norm(vec1)}")
print(f"vec2: {vec2}, norm: {np.linalg.norm(vec2)}")
print(f"vec3: {vec3}, norm: {np.linalg.norm(vec3)}")
print()

# Calculate expected values
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def l2_distance(a, b):
    return np.linalg.norm(a - b)

print("Expected similarity values:")
print(f"cosine_sim(vec1, vec1) = {cosine_similarity(vec1, vec1):.6f} (identical)")
print(f"cosine_sim(vec1, vec2) = {cosine_similarity(vec1, vec2):.6f} (very similar)")
print(f"cosine_sim(vec1, vec3) = {cosine_similarity(vec1, vec3):.6f} (orthogonal)")
print()

print("Expected L2 distances:")
l2_11 = l2_distance(vec1, vec1)
l2_12 = l2_distance(vec1, vec2)
l2_13 = l2_distance(vec1, vec3)
print(f"L2(vec1, vec1) = {l2_11:.6f}")
print(f"L2(vec1, vec2) = {l2_12:.6f}")
print(f"L2(vec1, vec3) = {l2_13:.6f}")
print()

print("Expected L2² (squared):")
print(f"L2²(vec1, vec1) = {l2_11**2:.6f}")
print(f"L2²(vec1, vec2) = {l2_12**2:.6f}")
print(f"L2²(vec1, vec3) = {l2_13**2:.6f}")
print()

# Relationship between cosine similarity and L2 distance for normalized vectors:
# L2² = 2 * (1 - cosine_similarity)
# Therefore: cosine_similarity = 1 - (L2² / 2)
print("Verification: cosine_sim = 1 - (L2² / 2) for normalized vectors:")
print(f"vec1-vec2: 1 - ({l2_12**2:.6f} / 2) = {1 - (l2_12**2 / 2):.6f} (expected: {cosine_similarity(vec1, vec2):.6f})")
print(f"vec1-vec3: 1 - ({l2_13**2:.6f} / 2) = {1 - (l2_13**2 / 2):.6f} (expected: {cosine_similarity(vec1, vec3):.6f})")
print()

# Now test with LanceDB
try:
    import lancedb
    import pyarrow as pa
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db = lancedb.connect(tmpdir)
        
        # Create table with 3D vectors
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 3)),
        ])
        
        table = db.create_table("test", schema=schema)
        
        # Add vectors
        table.add([
            {"id": "vec1", "vector": vec1.tolist()},
            {"id": "vec2", "vector": vec2.tolist()},
            {"id": "vec3", "vector": vec3.tolist()},
        ])
        
        print("=" * 80)
        print("LanceDB Results:")
        print("=" * 80)
        
        # Search for vec1 (should match itself perfectly)
        results = table.search(vec1.tolist()).limit(3).to_list()
        
        for row in results:
            rid = row["id"]
            dist = row["_distance"]
            
            # Try both formulas
            similarity_if_l2 = 1 - (dist * dist / 2)
            similarity_if_l2squared = 1 - (dist / 2)
            
            # Get expected cosine similarity
            if rid == "vec1":
                expected_cos = cosine_similarity(vec1, vec1)
            elif rid == "vec2":
                expected_cos = cosine_similarity(vec1, vec2)
            else:
                expected_cos = cosine_similarity(vec1, vec3)
            
            print(f"\nResult: {rid}")
            print(f"  _distance = {dist:.6f}")
            print(f"  If _distance is L2:  similarity = 1 - (dist² / 2) = {similarity_if_l2:.6f}")
            print(f"  If _distance is L2²: similarity = 1 - (dist / 2) = {similarity_if_l2squared:.6f}")
            print(f"  Expected cosine_sim = {expected_cos:.6f}")
            
            # Check which formula matches
            if abs(similarity_if_l2 - expected_cos) < 0.01:
                print(f"  ✓ _distance appears to be L2 distance")
            elif abs(similarity_if_l2squared - expected_cos) < 0.01:
                print(f"  ✓ _distance appears to be L2² (squared distance)")
            else:
                print(f"  ✗ Neither formula matches! Unexpected behavior.")

except ImportError:
    print("LanceDB not installed. Install with: pip install lancedb")
except Exception as e:
    print(f"Error testing LanceDB: {e}")
    import traceback
    traceback.print_exc()
