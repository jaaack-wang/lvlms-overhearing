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
@inproceedings{wang-etal-2025-lvlms,
    title = "{LVLM}s are Bad at Overhearing Human Referential Communication",
    author = "Wang, Zhengxiang  and
      Li, Weiling  and
      Kaliosis, Panagiotis  and
      Rambow, Owen  and
      Brennan, Susan",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.849/",
    doi = "10.18653/v1/2025.emnlp-main.849",
    pages = "16758--16782",
    ISBN = "979-8-89176-332-6"
}
```

