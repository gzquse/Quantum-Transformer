import itertools
from collections import defaultdict

def apply_qubit_permutation(bitstring, permutation):
    """Apply a qubit permutation to a bitstring"""
    # permutation[i] tells us which position in the original string 
    # should go to position i in the new string
    return ''.join(bitstring[permutation[i]] for i in range(len(bitstring)))

def reverse_permutation(counts, permutation):
    """Apply reverse permutation to counts dictionary"""
    new_counts = {}
    for bitstring, count in counts.items():
        new_bitstring = apply_qubit_permutation(bitstring, permutation)
        new_counts[new_bitstring] = count
    return new_counts

def compare_distributions(dist1, dist2):
    """Calculate similarity between two count distributions"""
    all_keys = set(dist1.keys()) | set(dist2.keys())
    total_diff = 0
    total_counts = 0
    
    for key in all_keys:
        count1 = dist1.get(key, 0)
        count2 = dist2.get(key, 0)
        total_diff += abs(count1 - count2)
        total_counts += count1 + count2
    
    # Return similarity score (higher is better)
    return 1 - (total_diff / total_counts)

def find_best_qubit_mapping(ionq_counts, local_counts, num_qubits=4):
    """Find the best qubit mapping between IonQ and local results"""
    
    best_score = 0
    best_permutation = None
    results = []
    
    # Test all possible permutations
    for perm in itertools.permutations(range(num_qubits)):
        # Apply permutation to IonQ data
        remapped_ionq = reverse_permutation(ionq_counts, perm)
        
        # Calculate similarity
        score = compare_distributions(remapped_ionq, local_counts)
        results.append((perm, score, remapped_ionq))
        
        if score > best_score:
            best_score = score
            best_permutation = perm
    
    return best_permutation, best_score, results

# Test with your first measurement
ionq_first = {'0000': 436, '1000': 387, '0100': 190, '1100': 796, '0010': 1171, '1010': 731, '0110': 925, '1110': 1303, '0001': 166, '1001': 375, '0101': 666, '1101': 127, '0011': 550, '1011': 1315, '0111': 482, '1111': 380}

local_first = {'0000': 824, '1000': 51, '0100': 1220, '1100': 634, '0010': 820, '1010': 351, '0110': 1013, '1110': 775, '0001': 431, '0101': 34, '1101': 592, '0011': 447, '1011': 833, '0111': 222, '1111': 508, '1001': 1245}

best_perm, best_score, all_results = find_best_qubit_mapping(ionq_first, local_first)

print(f"Best permutation: {best_perm}")
print(f"Best similarity score: {best_score:.3f}")
print(f"This means: qubit 0→{best_perm[0]}, qubit 1→{best_perm[1]}, qubit 2→{best_perm[2]}, qubit 3→{best_perm[3]}")

# Show top 5 results
print("\nTop 5 permutations:")
sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
for i, (perm, score, _) in enumerate(sorted_results[:5]):
    print(f"{i+1}. {perm} → score: {score:.3f}")


def test_common_mappings(ionq_counts, local_counts):
    """Test common qubit mapping patterns"""
    
    patterns = {
        "Identity": (0, 1, 2, 3),
        "Reverse": (3, 2, 1, 0),
        "Swap 0↔1": (1, 0, 2, 3),
        "Swap 2↔3": (0, 1, 3, 2),
        "Swap 0↔3": (3, 1, 2, 0),
        "Rotate left": (1, 2, 3, 0),
        "Rotate right": (3, 0, 1, 2),
    }
    
    results = {}
    for name, perm in patterns.items():
        remapped = reverse_permutation(ionq_counts, perm)
        score = compare_distributions(remapped, local_counts)
        results[name] = (score, remapped)
        print(f"{name:12} → similarity: {score:.3f}")
    
    return results

print("Testing common mapping patterns:")
pattern_results = test_common_mappings(ionq_first, local_first)

def validate_permutation_across_measurements(ionq_data, local_data, permutation):
    """Validate a permutation across all measurements"""
    
    scores = []
    for i, (ionq_counts, local_counts) in enumerate(zip(ionq_data, local_data)):
        # Convert local numpy format to regular int
        local_clean = {k: int(v) for k, v in local_counts.items()}
        
        remapped_ionq = reverse_permutation(ionq_counts, permutation)
        score = compare_distributions(remapped_ionq, local_clean)
        scores.append(score)
        
        if i < 3:  # Show first 3 as examples
            print(f"Measurement {i}: similarity = {score:.3f}")
    
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage similarity across all measurements: {avg_score:.3f}")
    return scores, avg_score

# Your data (first few measurements)
ionq_data = [
    {'0000': 436, '1000': 387, '0100': 190, '1100': 796, '0010': 1171, '1010': 731, '0110': 925, '1110': 1303, '0001': 166, '1001': 375, '0101': 666, '1101': 127, '0011': 550, '1011': 1315, '0111': 482, '1111': 380},
    {'0000': 235, '1000': 545, '0100': 292, '1100': 583, '0010': 1149, '1010': 1241, '0110': 652, '1110': 574, '0001': 467, '1001': 260, '0101': 311, '1101': 267, '0011': 473, '1011': 818, '0111': 907, '1111': 1226},
    {'0000': 296, '1000': 88, '0100': 402, '1100': 859, '0010': 227, '1010': 1261, '0110': 297, '1110': 751, '0001': 301, '1001': 937, '0101': 281, '1101': 121, '0011': 1473, '1011': 400, '0111': 1385, '1111': 921}
]

# Test the best permutation across multiple measurements
if best_perm:
    print(f"\nValidating permutation {best_perm} across measurements:")
    scores, avg = validate_permutation_across_measurements(
        ionq_data, 
        [local_first, local_first, local_first],  # You'd use your actual local data here
        best_perm
    )

def correct_all_ionq_data(ionq_data_list, permutation):
    """Apply the correction to all your IonQ measurements"""
    corrected_data = []
    
    for measurement in ionq_data_list:
        corrected = reverse_permutation(measurement, permutation)
        corrected_data.append(corrected)
    
    return corrected_data

# Apply correction to all your data
if best_perm:
    print(f"\nApplying permutation {best_perm} to correct IonQ data...")
    corrected_ionq_data = correct_all_ionq_data(ionq_data, best_perm)
    
    print("Before correction:", ionq_data[0])
    print("After correction: ", corrected_ionq_data[0])
    print("Local reference:  ", local_first)


import numpy as np

def calculate_entropy(counts_dict):
    """Calculate Shannon entropy of the distribution"""
    total_shots = sum(counts_dict.values())
    probabilities = [count/total_shots for count in counts_dict.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

local_entropy = calculate_entropy(ionq_first)
ionq_entropy = calculate_entropy(local_first)

print(f"Local entropy: {local_entropy:.3f} bits")
print(f"IonQ entropy: {ionq_entropy:.3f} bits") 
print(f"Maximum entropy (uniform): {np.log2(16):.3f} bits")

# Low entropy = concentrated distribution (local)
# High entropy = uniform distribution (IonQ)