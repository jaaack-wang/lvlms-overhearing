Data and code for our paper *LVLMs are Bad at Overhearing Human Referential Communication*, which is accepted to EMNLP 2025 (Main).



## Data

The exact data we used in our paper can be found in [this zip file](https://drive.google.com/file/d/1QX-MkVekARrVszltc7dzqcYc4fFAbFEf/view?usp=sharing). In this repo, we also include a zip file called `raw_data.zip`, which has everything needed to reproduce the data for our experiments. The major difference is that `raw_data.zip` does not have the input images, each of which contains 13 or 10 object pictures placed in a 3x5 or 2x5 grids. Instead, it only has individual pictures for all objected included in our corpus. See the two notebooks (`create_data.ipynb` and `create_image_playbooks.ipynb`) inside the `notebooks` folder for reference. To reproduce the exact data used in our paper, use the `mapper.json` file inside each image directory to place the right object pictures in the right order.

To protect our data from data contamintation, these two zip files are password protected. The password is the name of this repo. 

**To prevent data contamination, please do not directly upload the raw data of the corpus into the internet**. Thank you!



## Code

Install the packages in the `requirements.txt` and use the `experiments.py` to re-run our experiments. 

- Example for baskets:

```
python experiments.py --data_fp=data/baskets-matching-data.xlsx --image_dire=data/baskets-grid --setup_name="one transcript at a time" --rounds=1,2,3,4 --num_pairs=10 --models=openai/gpt-4o-mini-2024-07-18 --num_experiments_per_experiment=5 --output_fn=gpt-4o-mini
```

- Example for dogs:

```
python experiments.py --data_fp=data/dogs-matching-data.xlsx --image_dire=data/dogs-grid --setup_name="one transcript at a time" --rounds=1,2,3,4 --num_pairs=10 --models=openai/gpt-4o-mini-2024-07-18 --num_experiments_per_experiment=5 --output_fn=gpt-4o-mini
```

Use the flag `--rounds` to specify the rounds to run an LVLM on, such as from round 1 to round 4, round 2 to round 4 etc. 

See `examples_run.sh` for more examples of how to re-implement our experiments. 



## Citation 

```
@misc{wang2025lvlmsbadoverhearinghuman,
      title={LVLMs are Bad at Overhearing Human Referential Communication}, 
      author={Zhengxiang Wang and Weiling Li and Panagiotis Kaliosis and Owen Rambow and Susan E. Brennan},
      year={2025},
      eprint={2509.11514},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.11514}, 
}
```

