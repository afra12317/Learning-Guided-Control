# TODO

## ✅ Completed
Run
```bash
   ros2 launch mppi_control mppi_example.py
   ```

---
## 📌 Next Steps todos
- [x] Add obstacle avoidance
- [x] Use GPU + Cache to accelerate code
- [ ] Add a warm start
- [ ] Tune on the vehicle

---

## 🚧 Current Issues in the Code
1. **Missing reference speed** – there should be a reference velocity, and the reward should include a `vel_cost` term to encourage speed tracking.

2. Detect collisions between waypoints (i.e., trajectory segments), not just at individual states.

---