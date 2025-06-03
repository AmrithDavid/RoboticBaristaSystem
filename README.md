# Robotic Barista System

An AI-driven robotic barista system that uses computer vision, reinforcement learning, and robotics to automate drink preparation.

## Project Overview

This project implements a complete AI robotic system that:
1. Uses a MobileNetV2 CNN to classify drink types (coffee/matcha) from images
2. Controls a simulated robot arm in PyBullet using reinforcement learning
3. Creates an intelligent agent to make decisions and complete the drink preparation process

## Components

- **Perception System**: MobileNetV2-based classifier for drink type recognition
- **Robotic Arm**: Simulated in PyBullet physics engine
- **Reinforcement Learning**: PPO-based agent that learns to control the arm
- **Decision Agent**: Coordinates the overall preparation process

## Requirements

See `requirements.txt` for a full list of dependencies.

## Structure

- `models/` - Neural network models for drink classification
- `robotic_arm/` - PyBullet environment and robot control
- `reinforcement/` - Reinforcement learning implementation
- `decision/` - Decision-making agent
- `data/` - Training data and resources

## Group Information
### Group 15 - Coffee
Amrith David (14266415)
Rachel Nguyen (141371196)
Shing Hin Yuen (24544963)
Zubayr Parker (14268231)
John Lim (12050326)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Implementation

Run the complete system from inputted images in a random order.

```bash
python main.py # default
```

### Parameters

| argument | Description | Default |
| ----- | ----- | ----- |
| `--model` | Path to trained perception model | "models/final_model.pth" |
| `--images-dir` | Directory containing drink images | "data/true_test" |
| `--orders` | Number of random orders to process | 2 |
| `--keep-open` | Keep simulation window open after processing | |

### Examples

```bash
python main.py --model models\final_model.pth --images-dir data\true_test --orders 2 --keep-open

python main.py --model models\best_model.pth --images-dir data\test\matcha --orders 3
python main.py --model models\final_model.pth --images-dir data\test\coffee --orders 4
```
