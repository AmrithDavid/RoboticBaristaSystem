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

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete system
python main.py
