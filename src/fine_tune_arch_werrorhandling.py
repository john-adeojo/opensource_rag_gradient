import json
import pathlib
from dotenv import load_dotenv
import setup
import os
from gradientai import Gradient

# Set Environment Variables
setup.set_environment_variables()

# set script fine-tuning variables
data = "additional_yoda_style_questions_responses_thousand.jsonl"
rank = 17
num_epochs = 4
learning_rate = 1e-5

def read_samples_from_jsonl_in_batches(file_name, batch_size=100):
    # Construct the file path relative to the parent of the current script's directory
    current_dir = pathlib.Path(__file__).parent
    file_path = current_dir.parent / file_name

    batch = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            batch.append(data)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def save_state(epoch, batch_number, model_id):
    state = {'epoch': epoch, 'batch_number': batch_number, 'model_id': model_id}
    with open('finetune_state.json', 'w') as file:
        json.dump(state, file)

def load_state():
    if os.path.exists('finetune_state.json'):
        with open('finetune_state.json', 'r') as file:
            return json.load(file)
    return {'epoch': 0, 'batch_number': 0}

def example_query_yoda():
    query = """  
        Instruction:\nWhat's the capital of England?\n\n### Response:\n{response}</s> 
    """
    return query

def main():
    load_dotenv()
    state = load_state()

    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    model_id = state.get('model_id')
    if model_id:
        print(f"Resuming fine-tuning on model: {model_id}")
        new_model_adapter = gradient.get_model_adapter(model_id)
    else:
        new_model_adapter = base_model.create_model_adapter(
            name="nous-hermes2-talk-yoda",
            rank= rank,
            learning_rate= learning_rate,
        )
        model_id = new_model_adapter.id
        print(f"Created new model adapter with id {model_id}")

    sample_query = example_query_yoda()

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
    print(f"Generated before finetuning!: {completion}")

    start_epoch = state.get('epoch', 0)
    start_batch = state.get('batch_number', 0)

    for epoch in range(start_epoch, num_epochs):
        batch_number = 0
        for batch in read_samples_from_jsonl_in_batches(data):
            batch_number += 1
            if epoch == start_epoch and batch_number <= start_batch:
                continue

            try:
                print(f"Processing batch {batch_number} in epoch {epoch + 1}")
                new_model_adapter.fine_tune(samples=batch)
                save_state(epoch, batch_number, model_id)
            except Exception as e:
                print(f"Error occurred: {e}")
                return

        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
        print(f"Generated after epoch {epoch + 1}: {completion}")

    print("Finetuning complete!")
    gradient.close()

if __name__ == "__main__":
    main()
