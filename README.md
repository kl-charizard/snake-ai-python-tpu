# Snake AI Reinforcement Learning (with TPU on colab)

**TPU Efficiency:** TPU acceleration shines for large-scale tensor computations. Although this game code uses neural network training, the bulk of the code (game logic) is not accelerated by TPU. You may not see huge performance gains, but this setup is useful for experimenting with TPU in reinforcement learning contexts.

## Instructions for Running on Google Colab from GitHub

1. **Open a New Google Colab Notebook**  
   Go to [Google Colab](https://colab.research.google.com/).

2. **Enable TPU Runtime**  
   - Click on **Runtime** in the menu, then **Change runtime type**.
   - Under **Hardware accelerator**, select **TPU**.
   - Click **Save**.

3. **Clone Your GitHub Repository**  
   In a Colab cell, run:
   ```python
   !git clone https://github.com/kl-charizard/snake-ai-python-tpu
   %cd snake-ai-python-tpu
   ```

4. **Install torch_xla**  
   In a new cell, run:
   ```python
   !pip install torch-xla
   ```
   (Colab’s TPU runtime should already have compatible versions of PyTorch and torch_xla. If needed, restart the runtime after installation.)

5. **Run Your Script**  
   To start training (which is the default mode), run:
   ```python
   !python3 snake_tpu.py
   ```
   For demo mode (note that GUI rendering is disabled), run:
   ```python
   !python3 snake_tpu.py demo
   ```

6. **Monitor the Output**  
   Training progress (game count, score, record) will be printed in the notebook’s output cell.

---

**Additional Notes:**

- **Rendering:** Since TPU environments (and Colab TPU sessions) don’t support graphical display, the `render` option is set to `False`. If you want to test GUI locally (on your own machine with a GPU/CPU), you can change the parameter when instantiating `SnakeGameAI`.
- **Saving/Loading Models:** The model is saved locally (in the Colab session’s file system). You might consider additional steps to upload your model to Google Drive or GitHub if you wish to preserve training progress.

