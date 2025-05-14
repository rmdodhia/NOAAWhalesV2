import random
from collections import defaultdict

# Mapping of each location to its species (can be multiple)
location_species_map = {
    '201D': 'beluga', '206D': 'beluga', '213D': 'beluga', '214D': 'beluga',
    '215D': 'beluga', '216D': 'beluga', '218D': 'beluga',
    'ALBS04': 'humpback', 'ALNM01': 'humpback', 'Chinitna': ['humpback', 'killerwhale'],
    'Iniskin': ['humpback', 'killerwhale'], 'PtGraham': ['humpback', 'killerwhale'],
    'SWCorner': 'killerwhale'
}

def generate_species_balanced_folds(
    location_species_map,
    num_folds=3,
    test_size=2,
    val_size=1,
    max_attempts=1000
):
    all_locations = list(location_species_map.keys())
    folds = []
    attempts = 0
    seen_splits = set()

    while len(folds) < num_folds and attempts < max_attempts:
        attempts += 1
        random.shuffle(all_locations)

        # Sample test and val locations
        test = random.sample(all_locations, test_size)
        remaining = [loc for loc in all_locations if loc not in test]
        val = random.sample(remaining, val_size)
        train = [loc for loc in all_locations if loc not in test + val]

        # Make sure we haven't used this exact split before
        split_key = tuple(sorted(test + val))
        if split_key in seen_splits:
            continue
        seen_splits.add(split_key)

        # Check species presence in test + val
        test_val_species = set()
        for loc in test + val:
            species = location_species_map[loc]
            if isinstance(species, list):
                test_val_species.update(species)
            else:
                test_val_species.add(species)

        # Check species presence in train
        train_species = set()
        for loc in train:
            species = location_species_map[loc]
            if isinstance(species, list):
                train_species.update(species)
            else:
                train_species.add(species)

        # Only accept folds with all 3 species in both train and test+val
        if len(test_val_species) == 3 and len(train_species) == 3:
            folds.append({
                'test': test,
                'val': val,
                'train': train
            })

    return folds

# Example usage
if __name__ == "__main__":
    folds = generate_species_balanced_folds(location_species_map, num_folds=3)
    for i, fold in enumerate(folds):
        print(f"\n--- Fold {i+1} ---")
        print(f"Test:  {fold['test']}")
        print(f"Val:   {fold['val']}")
        print(f"Train: {fold['train']}")
