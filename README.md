# CS4186 Integrated Project

Name: HE Yu, ZENG Baoyi
Student ID: 72400334,
Contributions:

## Environment and data
Windows CPU run: PyTorch 2.11.0+cpu, NumPy 2.4.3. Install dependencies in your Python environment:

    python -m pip install -r requirements.txt

Place the original course-provided `Tutorial_08_CV_Latent_Models.zip` beside the scripts. The data loader finds `mnist_binary_subset.npz` inside the ZIP, or in the extracted `Tutorial_08_CV_Latent_Models/data/` or `data/` directory. The dataset is intentionally not duplicated here. Preserve the supplied splits.

## Run
From the extracted submission directory, train from random initialization and evaluate:

    python CS4186_Integrated_Project_Windows.py

Alternatively open `CS4186_Integrated_Project_Windows.ipynb` in JupyterLab and run all cells.

To evaluate the supplied trained checkpoint without retraining:

    python evaluate.py

The evaluation script uses the same fixed configuration and loads `project_outputs/vae.pt`. It regenerates the MSE table and evaluation figures, leaving the training log and training curves intact. Both commands write into `project_outputs/`; back up the supplied original results before rerunning. Small numerical differences across library versions or platforms are possible. The separate evaluation entry point is derived from the training script and syntax-checked, not independently executed here.

## Contents
`CS4186_Report.pdf` is the four-page report. `project_outputs/` contains the original Windows run checkpoint, configuration and seeds, training log, eight-row MSE table, and eight figures. The Python script and notebook contain the training and evaluation implementation.

## Sources and disclosure
Course Tutorial 08 `coding/vision_lab.py`: ConvVAE architecture reused; training logging and checkpoint selection adapted. Gaussian/Sobel processing, reparameterization, per-image BCE and analytic KL, fixed-route evaluation and visualizations are implemented in the project code. ChatGPT assisted with code drafting, Windows debugging, result interpretation and report preparation. Review all code and conclusions, complete author/contribution details and confirm disclosure requirements under the course policy before submission.
