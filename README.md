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

## 🤔 Unclear Points(by Haiyue)

1. **Target Cost Function from Proposal:**

   $$ 
    \ell(x_t,u(x_t)) = (x_{t+1} - \hat{x}_{t+1})^\top Q_f (x_{t+1} - \hat{x}_{t+1}) + a \mathbf{1}(x_{t+1} \in \mathcal{X}_{\text{unsafe}}) \\
    \text{s.t. } x_{t+1} = f(x_t, u_t)
    $$

   **Question:** How do we determine the **unsafe region** $\mathcal{X}_{\text{unsafe}}$?
     - Is it directly provided by an RL model?
     - Or do we need to perform grid-based obstacle checking like in RRT?

2. **Control Input vs. Trajectory Planning:**
   - Should we provide direct **control inputs** $u(t)$ to MPPI?
   - Or is it acceptable to generate a **sequence of waypoints** and let MPPI track that trajectory?
   - From my understanding (though I might be misunderstanding MPPI), it seems reasonable that if the decision layer outputs a sequence of waypoints forming a trajectory, then MPPI can simply track that trajectory(and avoid obstacles at the same time).

---

## 🐞 Current Code Issues

1. **Very low speed** when running PP + MPPI:
   - The velocity computed by MPPI is significantly lower than the maximum speed I set.
   - idk why
   
2. **Slow performance** overall
    -  Code optimization is needed in later stages to improve runtime performance.

---
## 📌 Next Steps
