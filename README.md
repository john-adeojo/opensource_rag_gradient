
# About This Repo

This repository accompanies the technical blog: [placeholder for link] and the YouTube video: [Placeholder for link]

## Pre-requisites
You need to sign up for Gradient to run the fine-tuning script: [placeholder for link]

## Getting Set Up
Clone this repository to a local location:
```
gh repo clone john-adeojo/opensource_rag_gradient
```

Then create a virtual environment and install the requirements:

### Create the virtual environment: 
```
conda create -n <your_env> python=3.10 pip
conda activate <your_env>
```

### Install the requirements:
```
pip install -r requirements.txt
```

After activating your virtual environment, open the `configurations.json` file and enter your `access_token` and `workspace_id` from Gradient.

## Running the Fine-Tuning
By default, the fine-tuning will run on the Nous-Hermes2 model using the `additional_yoda_style_questions_responses_thousand.jsonl` data. You can modify this by adjusting the script. 

Run the script with:
```
python fine_tune.py run
```

## Working with the UI

Change the `model_adapter_id` at the top of the `opensource_rag_gradient/src/chat.py` file to match the fine-tuned model you've created on Gradient.

You can also work with the base model already included in the script. Note that the Llama_2_chat model hasn't been set up with its template, so it might not work well.

To initiate the Chainlit UI, run:
```
chainlit run chat.py
```

Once the UI opens in a web browser, go to the settings menu and select a model.

Index a Wikipedia page by providing the EXACT name of the page. For example, to index the page https://en.wikipedia.org/wiki/AI_safety, enter "AI safety" in the input box.

![Chainlit UI](https://github.com/john-adeojo/opensource_rag_gradient/blob/main/Images/Chainlit%20UI.png)

Wait for the confirmation message that the page has been indexed.

![Chainlit UI Confirmed](https://github.com/john-adeojo/opensource_rag_gradient/blob/main/Images/Chainlit%20UI%20confirmed.png)

Follow the same steps to index another page.

## For AI Development
