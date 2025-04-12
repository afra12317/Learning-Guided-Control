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

## 🐞 Current Code Issues

1. **Very low speed** when running PP + MPPI:
   - The velocity computed by MPPI is significantly lower than the maximum speed I set.
   - idk why
   
2. **Slow performance** overall
    -  Code optimization is needed in later stages to improve runtime performance.

---
## 📌 Next Steps todos
- [ ] Add obstacle avoidance
- [ ] Tune on the vehicle
- [ ] Improve running speed