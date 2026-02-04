# Embodied AI

This repository contains research and notes on modern robotics frameworks and Embodied AI models, specifically focusing on **LeRobot** and **Vision-Language-Action (VLA)** architectures.


## 1. LeRobot: A Unified Framework for Robotics

**LeRobot** is an open-source framework developed by Hugging Face to provide a comprehensive set of tools for datasets, models, and simulation. Its primary mission is to **democratize and standardize** the robotics ecosystem.

### Traditional Control Stacks vs. LeRobot

| Feature | Traditional Modular Stack | LeRobot Approach |
| :--- | :--- | :--- |
| **Structure** | Sequential pipeline (Sensing → Planning → Control) | Unified/End-to-End |
| **Integration** | Separate modules for detection, IK, and path planning | Policy-based integration of all steps |
| **Logic** | Hand-coded algorithms and heuristics | Data-driven (Imitation/Reinforcement Learning) |
| **Input/Output** | Multi-stage data transformation | Sensor Data $\rightarrow$ Policy $\rightarrow$ Motor Command |

---

## 2. SmolVLA: Vision-Language-Action Architecture

**SmolVLA** is a state-of-the-art model that bridges the gap between digital reasoning and physical interaction.

### What makes it a VLA?
According to the research, a VLA is essentially a **VLM (Vision-Language Model) + an Action Expert**.

* **Sensorimotor States:** It processes multi-camera RGB feeds alongside the robot's internal states (like joint angles).
* **Action Expert:** This module translates high-level semantic features (e.g., "The cup is on the right") into low-level motor commands (velocity, torque).
* **Continuous Actions:** Unlike discrete text tokens, it produces smooth, real-valued outputs for fluid movement.

### Comparative Overview

| Model Type | Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| **LLM** | Text | Text | Digital reasoning & dialogue |
| **Vision** | Images | Labels / Segments | Spatial understanding |
| **VLA** | Image + Text + State | **Robot Actions** | Physical Embodiment & Interaction |

> **The "Action" Factor:** For AI to move from the digital world to the physical world, it needs "Embodiment." VLA models enable this by converting textual instructions and visual perception directly into robotic motion.

---

## Technical Context
* **Simulation Environments:** MuJoCo, Gazebo.
* **Frameworks:** ROS 2, PyTorch, LeRobot.
* **Robotic Platforms:** UR3e, Panda, Trossen VX300s.


------------------


TASK 2
------------------
xxxxxx

replace the lines (46-49) with the following code:

    self.target_position = Point()
    self.target_position.x = 0.5
    self.target_position.y = 0.0
    self.target_position.z = 0.0

    --------- or ---------------

    self.target_position = Point()
    self.target_position.x = 1.0
    self.target_position.y = 1.0
    self.target_position.z = 0.0
