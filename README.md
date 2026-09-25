## Think-beyond-Instances

## ⚙️ Requirements
You can run pip install -r requirements.txt to deploy the environment.

## ⚙️ Data Preparation
1.  **Data Splitting:** In the experiments, we maintain the same data splitting scheme as the benchmarks.
2.  **Dataset:**  For all datasets (ACM23,AIME25 and so on), please download from the official source.

 ### 📂 Data Preparation

Think-beyond-Instances expects data in **JSONL format**.


## 🔧 Evaluation

To evaluate the predicted answer, run the following command:

```bash
conda create -n Think_beyond_Instances python=3.10
conda activate Think_beyond_Instances
cd code
python MATH500.py
