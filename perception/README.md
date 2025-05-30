# Perception System

## Perception Model Training

```bash
python perception\train_model.py --data-dir data --output-dir models\train_model.pth --epochs 100 --batch-size 50
```

## Perception System Test

```bash
# Displaying Images with its confidence
python perception\test_directory.py --model models\best_model.pth --directory data\train\matcha
python perception\test_directory.py --model models\final_model.pth --directory data\test\coffee

# Displaying confidence distribition & confusion matrix
python perception\evaluate_classifier.py --model models\final_model.pth --test-dir data\test
```

## Demostration

```bash
python perception\integrate.py --keep-open --model models\final_model.pth --images data\true_test\coffee_101.jpeg data\true_test\matcha_101.jpeg
```
