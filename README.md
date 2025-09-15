
<div align="center">
  <img src="./image/lads.jpg" alt="Logo" width="200">
  <h1 align="center">LightAutoDS-Tab</h1>
</div>

<div align="center">

<img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python3.10">
<a href="https://github.com/sb-ai-lab/LADS/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-BSD--3--Clause-green"></a>

</div>

![architecture](./image/architecture.png)

**LightAutoDS-Tab**, a multi-AutoML agentic system for tasks with tabular data, which combines an LLM-based code generation with several AutoML tools.

## ✨ Demo
[Watch the Video](https://www.youtube.com/watch?v=5e8eADd_HWE)

## 🚀 Quick Start

**Step 1: Clone the repository**

```shell
git clone https://github.com/sb-ai-lab/LADS.git
cd LADS
```

**Step 2: Create conda environment**

```shell
conda env create -f environment.yml
conda activate LightAutoDS
```

**Step 3. Set up environment variables**

You need to create a `.env` file in the root directory of the project.

```shell
cp .env_example .env
```

You will need to fill in the required API keys and other environment variables in the `.env` file.

You can also change some parameters in [`config.yml`](./config.yml).

**Step 4: Run the application**

```shell
streamlit run app.py
```

Your application will be hosted on [http://localhost:8501](http://localhost:8501) by default.

## Result
We evaluated our framework on eight Kaggle ML datasets and compared it with two state-of-the-art open-source solutions: AutoKaggle and AIDE.

To ensure consistency across competitions, we use the Normalized Performance Score (NPS). This score standardizes the results, with a higher value indicating better performance. 
$$
\begin{equation}
\mathrm{NPS}= \begin{cases}\frac{1}{1+s}, & \text { if } s \text { is smaller the better } \\ s, & \text { otherwise }\end{cases}
\end{equation}
$$

<img src="./image/table1.png" align="center">
<img src="./image/table2.png" width="50%" align="center">

## 📜 License

Distributed under the BSD 3-Clause License. See [`LICENSE`](./LICENSE) for more information.