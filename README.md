# TODO

## ✅ Completed

1. Now able to successfully run the MPPI example using the command:
   ```bash
   ros2 launch mppi_control mppi_example.py
   ```

2. Integrated Pure Pursuit with MPPI:
    - The Pure Pursuit implementation is a very naive version I wrote in Python, with a fixed lookahead and no extra features — mainly used for testing MPPI when RL is not involved.
   - Currently, Pure Pursuit outputs a sequence of waypoints ahead of the car.
   - These waypoints are used as a reference trajectory for MPPI to track.
   - Run using:
     ```bash
     ros2 launch mppi_control mppi_PP.py
     ```

---
## 📌 Next Steps todos
- [x] Add obstacle avoidance
- [ ] Use GPU + Cache to accelerate code
- [ ] Add a warm start
- [ ] Tune on the vehicle
- [ ] Improve running speed

---

## 🚧 Current Issues in the Code

1. **Unstable performance**.  
2. **Too slow** - Don't know how to optimize dynamically sized array。
3. **Missing reference speed** – there should be a reference velocity, and the reward should include a `vel_cost` term to encourage speed tracking.

---