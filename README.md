## Steps to reproduce
<details>
<summary>Step 1: Clone the main repository and the submodule</summary>

```shell
git pull --recurse submodules
```
</details>

<details>
<summary>Step 2: Create and activate virtual environment with conda</summary>

```shell
conda create -n env python=3.8.5
conda activate env
```
</details>

<details>
<summary>Step 3: Install the requriements</summary>

```shell
pip install -r requirements.txt
pip install -ve YOLOX
```
</details>

<details>
<summary>Step 4: Pull the dataset</summary>

```shell
dvc pull
```

</details>

<details>
<summary>Step 5: Create Weights & Biases account for visualisation </summary>
Sign up for Weights & Biases account <a href="https://wandb.ai/site">here</a>.
  
</details>

<details>
<summary>Step 6: Install wandb and login </summary>

```shell
pip install wandb
wandb login
```
</details>

<details>
<summary>Step 7: Train the model</summary>

```shell
./train.sh
```
</details>
