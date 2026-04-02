conda create --name robotouille python=3.9 -y
conda activate robotouille
pip install -r requirements.txt

pip install -e .
pip install -e agents/prompt_builder/gpt-cost-estimator