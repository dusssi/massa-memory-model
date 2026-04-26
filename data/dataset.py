import numpy as np


def generate_data(seq_len=15, num_samples=2000, num_classes=10, seed=42):
    """
    Memory-based sequence task.

    Task:
    Model must remember EITHER the first OR last element.

    This creates:
    - long-term dependency
    - no arithmetic confusion
    - clear advantage for attention/memory models
    """

    np.random.seed(seed)

    # Generate sequences
    X = np.random.randint(0, num_classes, size=(num_samples, seq_len))

    # Randomly choose whether to pick first or last
    selector = np.random.randint(0, 2, size=num_samples)

    y = np.where(selector == 0, X[:, 0], X[:, -1])

    return X, y


# Optional test
if __name__ == "__main__":
    X, y = generate_data()

    print("Sample Input:", X[0])
    print("Expected Output:", y[0])